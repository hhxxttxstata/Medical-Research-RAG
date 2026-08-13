"""
Rewrite Stage Trace — Step 2 诊断

统计（所有数字在触发 Rewrite Gate 的样本上）:
  1. Mean/Median Candidate Jaccard@10 (Original vs Fusion)
  2. Both-hit 31 题: Pre-rerank Gold rank 变化 (Original → Fusion)
  3. Post-rerank Gold rank 变化 (NoRewrite → Full)
  4. Final Top5/Top10: NoRewrite hit vs Full hit, Evicted, Rescued
  5. Rewrite-only candidate 进入 rerank 后排在 Gold 前面的次数
  6. 保留 NoRewrite 正确 → Full 错误的 case

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/rewrite_stage_trace.py

产出: eval_results/rewrite_stage_trace_<timestamp>.json
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

import numpy as np

from eval.test_questions import get_test_questions  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 10
FETCH_K = 20
RERANK_K = 10


def gold_rank(sources: list[dict], expected: str) -> int | None:
    """Gold 在结果中的 rank（0-based），不在则 None"""
    if not expected:
        return None
    expected_base = expected.rsplit(".", 1)[0]
    for i, s in enumerate(sources):
        fn = s["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            return i
    return None


def jaccard(a: list[dict], b: list[dict]) -> float:
    """候选 id 集合的 Jaccard 相似度"""
    sa = {r["id"] for r in a}
    sb = {r["id"] for r in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    print("=" * 70)
    print("  🔬 Rewrite Stage Trace (Step 2)")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)}）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    reranker = CrossEncoderReranker()

    def build_ret(enable_rewrite: bool):
        return Retriever(
            vector_store=store,
            embedding_provider=provider,
            top_k=TOP_K,
            generator=generator,
            enable_rewrite=enable_rewrite,
            enable_reranker=True,
            reranker=reranker,
            bm25_backend="disk",
            bm25_index_dir="lucene_bm25_index",
        )

    ret_norewrite = build_ret(False)
    ret_full = build_ret(True)

    questions = get_test_questions()
    print(f"  📝 测试题: {len(questions)}\n", flush=True)

    jaccards = []
    rank_changes = []  # Both-hit: Original→Fusion pre-rerank rank 变化
    post_rank_changes = []  # Both-hit: NoRewrite→Full post-rerank rank 变化
    stats = {
        "N_gate": 0,
        "NoRewrite_top5_hit": 0,
        "Full_top5_hit": 0,
        "NoRewrite_top10_hit": 0,
        "Full_top10_hit": 0,
        "Evicted_top5": 0,
        "Rescued_top5": 0,
        "Evicted_top10": 0,
        "Rescued_top10": 0,
        "rewrite_only_before_gold": 0,
    }
    both_hit_cases = []
    norew_correct_full_wrong = []  # NoRewrite 正确 → Full 错误
    full_correct_norew_wrong = []  # 反向

    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")

        # 触发 Gate 才统计
        if not (ret_full._can_rewrite() and ret_full._rewrite_gate(question)):
            continue
        stats["N_gate"] += 1

        # NoRewrite 路径（original only）
        nr_results = ret_norewrite.retrieve(question, top_k=TOP_K)
        # Full 路径（original + rewrites + 二次RRF + rerank）
        fl_results = ret_full.retrieve(question, top_k=TOP_K)

        # candidates 对比（rerank 前）：手动复刻取 candidates
        orig_cands = ret_norewrite._hybrid_retrieve(question, fetch_k=FETCH_K)[:RERANK_K]
        fl_cands = None
        rewritten = ret_full._rewrite_query(question)
        search_queries = ([question] + rewritten) if (rewritten and rewritten[0] != question) else [question]
        per_query = [ret_full._hybrid_retrieve(sq, fetch_k=FETCH_K) for sq in search_queries]
        if len(per_query) > 1:
            rrf_k = 60
            score_map = {}
            for q_idx, q_results in enumerate(per_query):
                weight = 1.5 if q_idx == 0 else 1.0
                for rank, r in enumerate(q_results):
                    cid = r["id"]
                    score_map.setdefault(cid, {"score": 0.0, "result": r})
                    score_map[cid]["score"] += weight / (rrf_k + rank + 1)
            fl_cands = [v["result"] for _, v in sorted(score_map.items(), key=lambda x: x[1]["score"], reverse=True)][
                :RERANK_K
            ]
        else:
            fl_cands = per_query[0][:RERANK_K]

        jaccards.append(jaccard(orig_cands, fl_cands))

        # Gold rank 对比
        o_rank = gold_rank(orig_cands, expected)
        f_rank = gold_rank(fl_cands, expected)
        nr_rank = gold_rank(nr_results, expected)
        fl_rank = gold_rank(fl_results, expected)

        # Both-hit: rank 变化
        if o_rank is not None and f_rank is not None:
            rank_changes.append(f_rank - o_rank)
            both_hit_cases.append({"question": question, "o_rank": o_rank, "f_rank": f_rank})
        if nr_rank is not None and fl_rank is not None:
            post_rank_changes.append(fl_rank - nr_rank)

        # Top5 / Top10 hit 对比
        nr5, fl5 = gold_rank(nr_results[:5], expected) is not None, gold_rank(fl_results[:5], expected) is not None
        nr10, fl10 = gold_rank(nr_results[:10], expected) is not None, gold_rank(fl_results[:10], expected) is not None
        stats["NoRewrite_top5_hit"] += nr5
        stats["Full_top5_hit"] += fl5
        stats["NoRewrite_top10_hit"] += nr10
        stats["Full_top10_hit"] += fl10
        if nr5 and not fl5:
            stats["Evicted_top5"] += 1
        if not nr5 and fl5:
            stats["Rescued_top5"] += 1
        if nr10 and not fl10:
            stats["Evicted_top10"] += 1
        if not nr10 and fl10:
            stats["Rescued_top10"] += 1

        # Rewrite-only candidate 在 rerank 后排在 Gold 前
        if fl_rank is not None and f_rank is not None:
            fl_ids = {r["id"] for r in fl_cands}
            orig_ids = {r["id"] for r in orig_cands}
            rewrite_only_before = sum(1 for r in fl_results[:fl_rank] if r["id"] in (fl_ids - orig_ids))
            stats["rewrite_only_before_gold"] += rewrite_only_before

        # 错误 case 定位
        if nr10 and not fl10:
            norew_correct_full_wrong.append({"question": question, "expected": expected})
        if not nr10 and fl10:
            full_correct_norew_wrong.append({"question": question, "expected": expected})

        if stats["N_gate"] % 10 == 0:
            print(f"    ...{stats['N_gate']}/80 题", flush=True)

    elapsed = time.time() - t0

    # ── 输出 ──
    print("\n" + "=" * 70)
    print("  📊 Stage Trace 结果")
    print("=" * 70)
    s = stats
    print(f"  N_gate = {s['N_gate']}")
    print(f"  Mean Jaccard@10 = {np.mean(jaccards):.3f} | Median = {np.median(jaccards):.3f}")
    print(
        f"  Both-hit rank 变化 (Fusion-Original): mean = {np.mean(rank_changes):+.2f}, median = {np.median(rank_changes):+.0f}"
    )
    print(f"  Post-rerank rank 变化 (Full-NoRewrite): mean = {np.mean(post_rank_changes):+.2f}")
    print(
        f"  Final Top5:  NoRewrite={s['NoRewrite_top5_hit']} Full={s['Full_top5_hit']} Evicted={s['Evicted_top5']} Rescued={s['Rescued_top5']}"
    )
    print(
        f"  Final Top10: NoRewrite={s['NoRewrite_top10_hit']} Full={s['Full_top10_hit']} Evicted={s['Evicted_top10']} Rescued={s['Rescued_top10']}"
    )
    print(f"  Rewrite-only candidate 排 Gold 前次数 = {s['rewrite_only_before_gold']}")

    print(f"\n  📋 NoRewrite 正确 → Full 错误 ({len(norew_correct_full_wrong)} 题):")
    for c in norew_correct_full_wrong:
        print(f"    ❌ {c['question'][:55]}")
    print(f"  📋 Full 正确 → NoRewrite 错误 ({len(full_correct_norew_wrong)} 题):")
    for c in full_correct_norew_wrong:
        print(f"    ✅ {c['question'][:55]}")

    out = OUT_DIR / f"rewrite_stage_trace_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": s,
                "mean_jaccard": float(np.mean(jaccards)),
                "median_jaccard": float(np.median(jaccards)),
                "mean_rank_change": float(np.mean(rank_changes)) if rank_changes else None,
                "mean_post_rank_change": float(np.mean(post_rank_changes)) if post_rank_changes else None,
                "norew_correct_full_wrong": norew_correct_full_wrong,
                "full_correct_norew_wrong": full_correct_norew_wrong,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

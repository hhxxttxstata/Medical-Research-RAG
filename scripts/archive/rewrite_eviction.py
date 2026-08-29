"""
Rewrite 效应诊断 — Candidate Eviction / Rescue 统计

目标: 量化 Rewrite + 跨 Query 二次 RRF 的净效应
  Eviction Rate = Evicted / O_hit    (Original 命中但 Fusion 后掉出 Top10)

统计位置: candidates[:10]（Reranker 之前）
只统计触发 Rewrite Gate 的样本（未触发对诊断无意义）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/rewrite_eviction.py

产出: eval_results/rewrite_eviction_<timestamp>.json
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

from eval.test_questions import get_test_questions  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 10
FETCH_K = 20
RERANK_K = 10


def gold_hit(sources: list[dict], expected: str) -> bool:
    """Gold 是否在 top-K 中（与 evaluate.py 同逻辑）"""
    if not expected or not sources:
        return False
    expected_base = expected.rsplit(".", 1)[0]
    return any(
        expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
        for s in sources
    )


def main():
    print("=" * 70)
    print("  🔬 Rewrite 效应诊断（Candidate Eviction / Rescue）")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: rag_docs_c300_500 ({store.count()} chunks)", flush=True)

    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)} 文档）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()} 文档", flush=True)

    # 构造 retriever，但手动复刻 retrieve 内部流程以在 candidates 处统计
    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=TOP_K,
        generator=generator,
        enable_rewrite=True,
        enable_reranker=False,  # 统计点在 rerank 之前
        reranker=None,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )

    questions = get_test_questions()
    print(f"  📝 测试题: {len(questions)}\n", flush=True)

    stats = {
        "N_gate_triggered": 0,
        "O_hit": 0,  # Original Top10 命中 Gold
        "F_hit": 0,  # Fusion Top10 命中 Gold
        "Evicted": 0,  # Original 命中但 Fusion 掉出
        "Rescued": 0,  # Original 没命中但 Fusion 救回
        "Both_hit": 0,  # 都命中
    }
    detail_cases = []

    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")

        # 1. Rewrite Gate 判断
        needs_rewrite = retriever._can_rewrite() and retriever._rewrite_gate(question)

        # 2. Original 单独检索（与 fusion 同流程，仅单条 query）
        orig_results = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)

        o_hit = gold_hit(orig_results[:TOP_K], expected)

        if not needs_rewrite:
            # 未触发 Gate：Original 即最终候选（不统计进 N）
            continue

        stats["N_gate_triggered"] += 1

        # 3. Rewrite + 跨 Query 二次 RRF（复刻 retrieve 内部）
        rewritten = retriever._rewrite_query(question)
        if rewritten and rewritten[0] != question:
            search_queries = [question] + rewritten
        else:
            search_queries = [question]

        per_query_results = []
        for sq in search_queries:
            per_query_results.append(retriever._hybrid_retrieve(sq, fetch_k=FETCH_K))

        if len(per_query_results) > 1:
            rrf_k = 60
            score_map = {}
            for q_idx, q_results in enumerate(per_query_results):
                weight = 1.5 if q_idx == 0 else 1.0
                for rank, r in enumerate(q_results):
                    cid = r["id"]
                    score_map.setdefault(cid, {"score": 0.0, "result": r})
                    score_map[cid]["score"] += weight / (rrf_k + rank + 1)
            fused = [v["result"] for _, v in sorted(score_map.items(), key=lambda x: x[1]["score"], reverse=True)]
        else:
            fused = per_query_results[0]

        candidates = fused[:RERANK_K]  # ← 统计点
        f_hit = gold_hit(candidates, expected)

        if o_hit:
            stats["O_hit"] += 1
            if f_hit:
                stats["Both_hit"] += 1
            else:
                stats["Evicted"] += 1
                detail_cases.append({"question": question, "type": "evicted", "expected": expected})
        else:
            if f_hit:
                stats["Rescued"] += 1
                detail_cases.append({"question": question, "type": "rescued", "expected": expected})
            # 都没命中：不入统计

        if f_hit:
            stats["F_hit"] += 1

        if len(detail_cases) <= 10 or len([c for c in detail_cases]) % 10 == 0:
            pass

    elapsed = time.time() - t0

    # ── 输出结果 ──
    print("\n" + "=" * 70)
    print("  📊 Rewrite 效应诊断结果")
    print("=" * 70)
    s = stats
    eviction_rate = s["Evicted"] / s["O_hit"] if s["O_hit"] else 0
    rescue_rate = s["Rescued"] / s["N_gate_triggered"] if s["N_gate_triggered"] else 0
    net = s["F_hit"] - s["O_hit"]

    print(f"  Rewrite Gate 触发样本数 N        = {s['N_gate_triggered']}")
    print(f"  ① Original Top10 命中 Gold O_hit  = {s['O_hit']}")
    print(f"  ② Fusion Top10 命中 Gold F_hit    = {s['F_hit']}")
    print(f"  ③ Evicted (Orig命中但Fusion掉出)   = {s['Evicted']}")
    print(f"  ④ Rescued (Orig没中但Fusion救回)   = {s['Rescued']}")
    print(f"  ⑤ Both_hit                        = {s['Both_hit']}")
    print("  ─────────────────────────────────────")
    print(f"  ⚠️ Candidate Eviction Rate = {s['Evicted']} / {s['O_hit']} = {eviction_rate:.1%}")
    print(f"  🆘 Rescue Rate = {s['Rescued']} / {s['N_gate_triggered']} = {rescue_rate:.1%}")
    print(f"  📈 净效应 (F_hit - O_hit) = {net:+d}")

    print("\n  📋 Evicted / Rescued 明细:")
    for c in detail_cases[:12]:
        print(f"    [{c['type']:<8}] {c['question'][:50]}")

    out = OUT_DIR / f"rewrite_eviction_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "eviction_rate": eviction_rate,
                "detail_cases": detail_cases,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out}  (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

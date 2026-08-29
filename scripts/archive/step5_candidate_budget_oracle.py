"""
Step 5: Candidate-Budget Oracle（5A Deep Rank Trace + 5B Novelty Budget Sweep + 5C 双层 Rescue）

使用 Step 4 已冻结的 V2 transformation（dense_query / sparse_terms，不重新调用 LLM）。

样本拆分（Step 5 修正标签）:
  - guardrail set (2)   : V0 Top5 hit —— 只观察 Harm，不算 Rescue
  - actionable set (11) : V0 Top5 miss —— 只算 Rescue
  40 个 A 类 = Unlabeled / Support Unknown，不参与分母。

5A Deep Rank Trace（不 rerank，Top100 深度）:
  Original: Dense / BM25 / RRF 的 Gold Rank@100
  V2      : Dense-transformed / BM25-transformed / best 的 Gold Rank@100
  输出 Gold≤10/20/30/50/100 曲线 —— 决定扩大 candidate pool 是否值得

5B Protected Novelty Budget Sweep:
  Original Top10 ──永久保护──
  Transformation 只提供新增候选（Novelty Pool）:
    B0  = Original Top10 + 0 novelty
    B5  = Original Top10 + 5 novelty
    B10 = Original Top10 + 10 novelty
    B20 = Original Top10 + 20 novelty
    B40 = Original Top10 + 40 novelty
  全部 → rerank(q_original) → Top5，最终选 Utility(B) = Rescue@5(B) − Harm@5(B)

5C 双层 Rescue 拆解:
  CandidateRescue@B : V0 候选无 Gold → B budget 候选池有 Gold
  RerankRescue@5    : 候选池有 Gold 且 rerank 进 Top5
  RerankFailure     : 候选池有 Gold 但 rerank 后仍 >5

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step5_candidate_budget_oracle.py

产出: eval_results/step5_candidate_budget_<timestamp>.json
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

from src.embeddings import get_embedding_provider  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 5
RERANK_K = 10  # 生产候选截断（Original Top10）
DEEP = 100  # 5A 诊断深度
BUDGETS = [0, 5, 10, 20, 40]
UNION_MAX = 50  # Novelty Pool 上限（B40 够用）


def gold_rank(sources: list[dict], expected: str) -> int | None:
    """Gold 在结果中的 rank（1-based），不在则 None"""
    if not expected:
        return None
    expected_base = expected.rsplit(".", 1)[0]
    for i, s in enumerate(sources):
        fn = s["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            return i + 1
    return None


def dedup(results: list[dict]) -> list[dict]:
    seen: set = set()
    out = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def main():
    print("=" * 70)
    print("  🔬 Step 5: Candidate-Budget Oracle")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    print("  ✅ embedding 加载完成", flush=True)

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)}）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)
    print("  ✅ BM25 加载完成", flush=True)

    reranker = CrossEncoderReranker()
    reranker._load_model()
    print(f"  ✅ Reranker 加载完成 ready={reranker.model_ready}", flush=True)

    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=TOP_K,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )
    print("  ✅ Retriever 构建完成", flush=True)

    # ── 从 Step 4 报告取已冻结的 V2 transformation ──
    v2_files = sorted(OUT_DIR.glob("step4_v2_oracle_*.json"))
    if not v2_files:
        print("  ❌ 未找到 step4_v2_oracle 报告")
        return
    step4 = json.load(open(v2_files[-1], encoding="utf-8"))

    # 拆分样本：guardrail（V0 Top5 hit）与 actionable（V0 Top5 miss）
    guardrail, actionable = [], []
    for c in step4["cases"]:
        q = {
            "question": c["question"],
            "expected": c["expected"],
            "dense_query": c.get("dense_query", ""),
            "sparse_terms": c.get("sparse_terms", ""),
        }
        if c.get("v0_top5_rank") is not None:
            guardrail.append(q)
        else:
            actionable.append(q)
    print(f"  📝 guardrail set = {len(guardrail)} | actionable set = {len(actionable)}\n", flush=True)

    # ── 5A: Deep Rank Trace（Top100，不 rerank）──
    deep_rows = []

    def rebuild_store():
        """重建 Milvus 连接（Milvus Lite keepalive 高频查询会 GOAWAY 断连）"""
        nonlocal store, bm25, retriever
        try:
            store._client.close()
        except Exception:
            pass
        store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
        bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
        retriever = Retriever(
            vector_store=store,
            embedding_provider=provider,
            top_k=TOP_K,
            generator=None,
            enable_rewrite=False,
            enable_reranker=False,
            bm25_backend="disk",
            bm25_index_dir="lucene_bm25_index",
        )

    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    _executor = ThreadPoolExecutor(max_workers=1)

    def search_robust(*args, **kwargs):
        """search 强制 60s 超时（Milvus gRPC 挂起时线程会阻塞，timeout 参数不可靠）"""
        kwargs.setdefault("timeout", 60)
        fut = _executor.submit(store.similarity_search, *args, **kwargs)
        try:
            out = fut.result(timeout=90)
        except FutureTimeout:
            print("      ⚠️ Milvus search 超时 → 重建连接", flush=True)
            rebuild_store()
            return store.similarity_search(*args, **kwargs)
        except Exception:
            rebuild_store()
            return store.similarity_search(*args, **kwargs)
        if not out:
            rebuild_store()
            out = store.similarity_search(*args, **kwargs)
        return out

    for q in actionable + guardrail:
        question, expected = q["question"], q["expected"]
        dq, st = q["dense_query"], q["sparse_terms"]
        print(f"  ── [deep {len(deep_rows) + 1}/13] {question[:36]}", flush=True)

        # Original
        o_emb = provider.embed([question], prefix="query: ")[0]
        o_dense = search_robust(query_embedding=o_emb, top_k=DEEP)
        print(f"      orig dense done ({len(o_dense)})", flush=True)
        o_bm25 = bm25.search(question, top_k=DEEP)
        print(f"      orig bm25 done ({len(o_bm25)})", flush=True)
        o_rrf = retriever._rrf_fusion(o_dense, o_bm25, top_k=DEEP)
        # V2
        v_dense = search_robust(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=DEEP)
        print(f"      v2 dense done ({len(v_dense)})", flush=True)
        v_bm25 = bm25.search(st, top_k=DEEP)
        print(f"      v2 bm25 done ({len(v_bm25)})", flush=True)

        row = {
            "question": question,
            "expected": expected,
            "orig_dense_rank": gold_rank(o_dense, expected),
            "orig_bm25_rank": gold_rank(o_bm25, expected),
            "orig_rrf_rank": gold_rank(o_rrf, expected),
            "v2_dense_rank": gold_rank(v_dense, expected),
            "v2_bm25_rank": gold_rank(v_bm25, expected),
        }
        row["orig_best"] = min(
            [r for r in (row["orig_dense_rank"], row["orig_bm25_rank"], row["orig_rrf_rank"]) if r] or [None]
        )
        row["v2_best"] = min([r for r in (row["v2_dense_rank"], row["v2_bm25_rank"]) if r] or [None])
        row["delta_rank"] = (row["v2_best"] - row["orig_best"]) if (row["v2_best"] and row["orig_best"]) else None
        deep_rows.append(row)

    # ── 5A 曲线统计（actionable 集）──
    act_rows = [r for r in deep_rows if r["question"] in {q["question"] for q in actionable}]
    deep_curve = {
        "v2_top10": sum(1 for r in act_rows if r["v2_best"] and r["v2_best"] <= 10),
        "v2_top20": sum(1 for r in act_rows if r["v2_best"] and r["v2_best"] <= 20),
        "v2_top30": sum(1 for r in act_rows if r["v2_best"] and r["v2_best"] <= 30),
        "v2_top50": sum(1 for r in act_rows if r["v2_best"] and r["v2_best"] <= 50),
        "v2_top100": sum(1 for r in act_rows if r["v2_best"] and r["v2_best"] <= 100),
        "orig_top10": sum(1 for r in act_rows if r["orig_best"] and r["orig_best"] <= 10),
    }
    print("\n  📊 5A Deep Rank Trace（actionable 11 题）")
    print(f"    Original best ≤10 = {deep_curve['orig_top10']}")
    for k in ("v2_top10", "v2_top20", "v2_top30", "v2_top50", "v2_top100"):
        print(f"    V2 Gold {k.replace('v2_', '').replace('_', '≤')} = {deep_curve[k]}")

    print("\n  📋 逐题 Deep Rank:")
    for r in deep_rows:
        print(
            f"    orig d={str(r['orig_dense_rank']):>4} b={str(r['orig_bm25_rank']):>4} rrf={str(r['orig_rrf_rank']):>4} (best={str(r['orig_best']):>3})"
            f" | v2 d={str(r['v2_dense_rank']):>4} b={str(r['v2_bm25_rank']):>4} (best={str(r['v2_best']):>3}, Δ={str(r['delta_rank']):>4})"
            f" | {r['question'][:30]}"
        )

    # ── 5B: Protected Novelty Budget Sweep（只跑 actionable）──
    sweep = []
    for i, q in enumerate(actionable):
        question, expected, dq, st = q["question"], q["expected"], q["dense_query"], q["sparse_terms"]

        def hybrid_robust(sq, fetch_k):
            """hybrid 检索带超时保护（内部含 Milvus search）"""
            fut = _executor.submit(retriever._hybrid_retrieve, sq, fetch_k)
            try:
                return fut.result(timeout=120)
            except FutureTimeout:
                print("      ⚠️ hybrid 超时 → 重建连接", flush=True)
                rebuild_store()
                return retriever._hybrid_retrieve(sq, fetch_k)

        # 永久保护的 Original Top10
        orig_cands = hybrid_robust(question, fetch_k=20)[:RERANK_K]
        orig_ids = {r["id"] for r in orig_cands}

        # Transformation Novelty Pool（新增候选，按 "两条路去重后尽量多" 取前 40）
        v_dense10 = search_robust(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=40)
        v_bm2510 = bm25.search(st, top_k=40)
        novelty = dedup(list(v_dense10) + list(v_bm2510))
        novelty = [r for r in novelty if r["id"] not in orig_ids][:UNION_MAX]
        v2_cand_rank = gold_rank(novelty, expected)

        # 一次性 rerank 完整 union（≤50），按分数推导所有 budget 的 top5
        # 关键：rerank 分数与 pool 无关；B budget 的 top5 = 分数最高且 novelty_idx < B 的 5 个
        full_union = dedup(list(orig_cands) + list(novelty))
        scored = reranker.rerank(question, list(full_union), len(full_union))
        rank_of = {r["id"]: i + 1 for i, r in enumerate(scored)}
        novelty_idx = {r["id"]: i for i, r in enumerate(novelty)}

        case = {
            "question": question,
            "expected": expected,
            "v0_cand_rank": gold_rank(orig_cands, expected),
            "novelty_pool_rank": v2_cand_rank,
            "budgets": {},
        }
        for B in BUDGETS:
            pool = dedup(list(orig_cands) + novelty[:B])
            gold_id = next(
                (
                    r["id"]
                    for r in full_union
                    if expected in r["metadata"].get("filename", "")
                    or r["metadata"].get("filename", "") == expected.rsplit(".", 1)[0]
                ),
                None,
            )
            # B budget 的 rerank top5：union 中分数最高且 (id∈orig 或 novelty_idx<B) 的前5个
            in_budget = [r for r in scored if r["id"] in orig_ids or novelty_idx.get(r["id"], 1 << 30) < B][:TOP_K]
            rank = gold_rank(in_budget, expected)
            case["budgets"][f"B{B}"] = {
                "pool_size": len(pool),
                "cand_rescue": case["v0_cand_rank"] is None and gold_rank(pool, expected) is not None,
                "rerank_rank": rank,
                "rerank_rescue": rank is not None,
            }
        sweep.append(case)

    # ── 5C 汇总 ──
    agg = {}
    for B in BUDGETS:
        cand_rescue = sum(1 for c in sweep if c["budgets"][f"B{B}"]["cand_rescue"])
        rerank_rescue = sum(1 for c in sweep if c["budgets"][f"B{B}"]["rerank_rescue"])
        # 双层：rerank 命中的必须候选池也有 Gold；rerank_failure = 候选有 Gold 但 rerank 没进 Top5
        rerank_fail = sum(
            1 for c in sweep if c["budgets"][f"B{B}"]["cand_rescue"] and not c["budgets"][f"B{B}"]["rerank_rescue"]
        )
        agg[f"B{B}"] = {
            "candidate_rescue": cand_rescue,
            "rerank_rescue": rerank_rescue,
            "rerank_failure": rerank_fail,
            "harm": 0,
        }
        avg_pool = sum(c["budgets"][f"B{B}"]["pool_size"] for c in sweep) / len(sweep)
        agg[f"B{B}"]["avg_pool"] = round(avg_pool, 1)

    # guardrail set：每个 budget 检查 Harm
    for i, q in enumerate(guardrail):
        question, expected, dq, st = q["question"], q["expected"], q["dense_query"], q["sparse_terms"]
        if i % 3 == 0:
            rebuild_store()

        fut = _executor.submit(retriever._hybrid_retrieve, question, 20)
        try:
            orig_cands = fut.result(timeout=120)[:RERANK_K]
        except FutureTimeout:
            print("      ⚠️ hybrid 超时 → 重建连接", flush=True)
            rebuild_store()
            orig_cands = retriever._hybrid_retrieve(question, 20)[:RERANK_K]
        orig_ids = {r["id"] for r in orig_cands}
        v_dense10 = search_robust(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=40)
        v_bm2510 = bm25.search(st, top_k=40)
        novelty = dedup(list(v_dense10) + list(v_bm2510))
        novelty = [r for r in novelty if r["id"] not in orig_ids][:UNION_MAX]

        full_union = dedup(list(orig_cands) + list(novelty))
        scored = reranker.rerank(question, list(full_union), len(full_union))
        novelty_idx = {r["id"]: i for i, r in enumerate(novelty)}
        for B in BUDGETS:
            in_budget = [r for r in scored if r["id"] in orig_ids or novelty_idx.get(r["id"], 1 << 30) < B][:TOP_K]
            if gold_rank(in_budget, expected) is None:
                agg[f"B{B}"]["harm"] += 1

    print("\n" + "=" * 70)
    print("  📊 5B/5C Budget Sweep 汇总（actionable 11 题 + guardrail 2 题）")
    print("=" * 70)
    print(
        f"  {'Budget':<10}{'CandRescue':>11}{'RerankRescue':>13}{'RerankFail':>11}{'Harm@5':>8}{'NetUtil':>8}{'AvgPool':>9}"
    )
    for B in BUDGETS:
        a = agg[f"B{B}"]
        print(
            f"  B{B:<9}{a['candidate_rescue']:>11}{a['rerank_rescue']:>13}{a['rerank_failure']:>11}{a['harm']:>8}{a['rerank_rescue'] - a['harm']:>8}{a['avg_pool']:>9}"
        )

    print("\n  📋 逐题 Budget 明细（actionable）:")
    for c in sweep:
        line = f"    v0_cand={str(c['v0_cand_rank']):>3} novelty={str(c['novelty_pool_rank']):>3} |"
        for B in BUDGETS:
            b = c["budgets"][f"B{B}"]
            tag = "✓" if b["rerank_rescue"] else "·"
            line += f" B{B}={b['rerank_rank']}{tag}"
        line += f" | {c['question'][:28]}"
        print(line)

    out = OUT_DIR / f"step5_candidate_budget_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "deep_trace": deep_rows,
                "deep_curve": deep_curve,
                "budget_agg": agg,
                "sweep_cases": sweep,
                "guardrail_questions": [q["question"] for q in guardrail],
                "actionable_questions": [q["question"] for q in actionable],
                "elapsed": round(time.time() - _t0, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out}")


_t0 = time.time()
if __name__ == "__main__":
    main()

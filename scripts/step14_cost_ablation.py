"""
Step 14: Cost-aware Agentic Policy A/B —— Frozen Agentic v2 vs Cost-aware v2.1

研究问题：能否保持 v2 的 Agentic capability（Rescue/Harm/OOD/Completeness），
同时显著减少昂贵 LLM 调用（grader / policy）与延迟？

设计（来自 Step 11 + Step 12 证据）:
  Step 11:  −Grader ≈ full          → grader 很多时候信号冗余
  Step 12:  bh_ood_02               → 但信号冲突时 grader 不可删
  → v2.1 = Cheap Signal Gate（零 LLM）优先，仅 uncertainty/conflict 时调用 LLM

对比维度（能力 vs 成本）:
  能力（保持）:  Final Rescue / Harm / OOD Reject / False Abstain / Completeness / Hop Recall
  成本（下降）:  LLM Grader Calls / Avg LLM Calls / Avg latency / p95 latency / Timeout rate

观测字段（生产可监控）:
  policy_source: cheap_signal / llm / rule_fallback
  grader_called / grader_reason / fallback_used / operational_error

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step14_cost_ablation.py

产出: eval_results/step14_cost_ablation_<timestamp>.json
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

FETCH_K = 20
TOP_K = 5


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"avg": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(vals)
    n = len(s)
    return {
        "avg": round(sum(s) / n, 1),
        "p95": round(s[min(int(n * 0.95), n - 1)], 1),
        "max": round(s[-1], 1),
    }


def main():
    print("=" * 70)
    print("  💰 Step 14: Cost-aware Agentic Policy A/B（v2 vs v2.1）")
    print("=" * 70, flush=True)

    from eval.rescue_metrics import (
        compute_agent_capability_metrics,
        evidence_recall_at_k,
        hop_gold_ids,
    )
    from src.agentic_rag import AgenticRAG
    from src.cost_aware_agentic_rag import CostAwareAgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=5,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )
    generator = create_generator()
    agent_v2 = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    agent_v21 = CostAwareAgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    bench = json.load(open("tests/benchmark_multi_hop.json", encoding="utf-8"))["benchmark"]
    print(f"  📝 Dev Benchmark: {len(bench)} 题（与 Step 12/13 相同，冻结）", flush=True)

    cases = []
    t0 = time.time()
    for i, b in enumerate(bench, 1):
        question = b["question"]
        qtype = b["type"]
        all_gold = set()
        for hg in hop_gold_ids(b):
            all_gold |= hg
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{qtype}] {question[:40]}", flush=True)

        # ── V0 Fixed RAG（baseline comparator）──
        v0_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:FETCH_K]
        v0_top5 = reranker.rerank(question, list(v0_cands), TOP_K)
        v0_er = evidence_recall_at_k(v0_top5, all_gold)

        # ── Frozen v2 ──
        c0 = generator.call_counts()
        r2 = agent_v2.run(question, fetch_k=FETCH_K, verbose=False)
        c1 = generator.call_counts()
        v2_top5 = r2["sources"][:TOP_K]
        v2_er = evidence_recall_at_k(v2_top5, all_gold)

        # ── Cost-aware v2.1 ──
        c2 = generator.call_counts()
        r21 = agent_v21.run(question, fetch_k=FETCH_K, verbose=False)
        c3 = generator.call_counts()
        v21_top5 = r21["sources"][:TOP_K]
        v21_er = evidence_recall_at_k(v21_top5, all_gold)

        v2_llm = {k: c1[k] - c0[k] for k in c1}
        v21_llm = {k: c3[k] - c2[k] for k in c3}

        print(
            f"    V0_ER={v0_er:.2f} v2_ER={v2_er:.2f} v2.1_ER={v21_er:.2f} "
            f"v2.1_route={r21['route']} src={r21['observation']['policy_source']} "
            f"grader={r21['observation']['grader_called']} "
            f"llm(v2={v2_llm['total']}, v2.1={v21_llm['total']}) "
            f"lat={r21['elapsed']}s",
            flush=True,
        )

        cases.append(
            {
                "question": b,
                "v0_sources": v0_top5,
                "v2_sources": v2_top5,
                "v21_sources": v21_top5,
                "v2_route": r2["route"],
                "v21_route": r21["route"],
                "v2_answer": r2["answer"],
                "v21_answer": r21["answer"],
                "v2_abstained": r2["abstained"],
                "v21_abstained": r21["abstained"],
                "v2_elapsed": r2["elapsed"],
                "v21_elapsed": r21["elapsed"],
                "v2_llm_calls": v2_llm,
                "v21_llm_calls": v21_llm,
                "v21_obs": r21["observation"],
                "v0_evidence_recall": v0_er,
            }
        )

    elapsed = time.time() - t0

    # ── 能力指标（同一 evaluator，只换 sources/route）──
    def _cases_for(suffix: str) -> list[dict]:
        return [
            {
                "question": c["question"],
                "v0_sources": c["v0_sources"],
                "v1_sources": c[f"{suffix}_sources"],
                "v1_route": c[f"{suffix}_route"],
                "v1_answer": c[f"{suffix}_answer"],
                "v1_abstained": c[f"{suffix}_abstained"],
            }
            for c in cases
        ]

    m2 = compute_agent_capability_metrics(_cases_for("v2"))
    m21 = compute_agent_capability_metrics(_cases_for("v21"))

    # ── 成本指标 ──
    n = len(cases)
    lat2 = [c["v2_elapsed"] for c in cases]
    lat21 = [c["v21_elapsed"] for c in cases]
    grader21 = sum(1 for c in cases if c["v21_obs"]["grader_called"])
    grader21_pct = grader21 / n
    grader_reasons: dict[str, int] = {}
    for c in cases:
        r = c["v21_obs"]["grader_reason"] or "n/a"
        grader_reasons[r] = grader_reasons.get(r, 0) + 1
    policy_llm21 = sum(c["v21_obs"]["policy_llm_calls"] for c in cases)
    fallback21 = sum(1 for c in cases if c["v21_obs"]["fallback_used"])
    op_err21 = sum(1 for c in cases if c["v21_obs"]["operational_error"] != "none")
    llm_calls2 = sum(c["v2_llm_calls"]["total"] for c in cases)
    llm_calls21 = sum(c["v21_llm_calls"]["total"] for c in cases)
    retr2 = sum(len([a for a in c["v2_route"] if a in ("RETRIEVE", "DECOMPOSE")]) for c in cases)
    retr21 = sum(c["v21_obs"]["retrieval_calls"] for c in cases)

    # ── 报告 ──
    def _cap(m):
        return {
            "Final Answer Acc": m["final_answer_accuracy"],
            "Evidence Recall@5": m["evidence_recall"],
            "Hop Recall@5": m["hop_recall"],
            "Completeness": m["completeness"],
            "Final Rescue": m["final_rescue"],
            "Harm": m["harm"],
            "NetUtility": m["net_utility"],
            "OOD Reject": m["ood_reject"],
            "False Abstain": m["false_abstain"],
            "Policy Action Acc": m["policy_action_accuracy"],
            "Avg Iterations": m["avg_iterations"],
        }

    print("\n" + "=" * 70)
    print("  📊 A/B: Frozen v2  vs  Cost-aware v2.1")
    print("=" * 70)
    cap2, cap21 = _cap(m2), _cap(m21)
    print(f"  {'能力维度':<20}{'v2':>12}{'v2.1':>12}")
    print(f"  {'-' * 46}")
    for k in cap2:
        print(f"  {k:<20}{str(cap2[k]):>12}{str(cap21[k]):>12}")

    print(f"\n  {'成本维度':<20}{'v2':>12}{'v2.1':>12}{'变化':>10}")
    print(f"  {'-' * 56}")
    rows = [
        (
            "LLM Grader Calls",
            f"{n}/16",
            f"{grader21}/{n}",
            f"-{(1 - grader21_pct):.0%}".replace("-0%", "0%") if grader21_pct < 0.5 else "",
        ),
        ("Avg LLM Calls/题", f"{llm_calls2 / n:.2f}", f"{llm_calls21 / n:.2f}", ""),
        ("Avg Latency (s)", f"{_stats(lat2)['avg']:.1f}", f"{_stats(lat21)['avg']:.1f}", ""),
        ("p95 Latency (s)", f"{_stats(lat2)['p95']:.1f}", f"{_stats(lat21)['p95']:.1f}", ""),
        ("Timeout Rate", "0", f"{op_err21}/{n}", ""),
    ]
    for label, v2v, v21v, delta in rows:
        print(f"  {label:<20}{v2v:>12}{v21v:>12}{delta:>10}")

    # 成本降幅（grader 调用为主指标）
    grader_reduction = (1 - grader21_pct) * 100
    latency_reduction = (1 - _stats(lat21)["avg"] / max(_stats(lat2)["avg"], 1e-9)) * 100
    print(f"\n  📉 Grader 调用降幅: {grader_reduction:.0f}%   Latency 降幅: {latency_reduction:.0f}%")

    # ── Grader 调用原因分布 ──
    print("\n  ── Grader 调用原因分布（v2.1）──")
    for r, cnt in sorted(grader_reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:<14}{cnt:>3} 题")

    # ── 能力保持 vs 成本下降逐题 ──
    print("\n  ── 逐题（route / grader / policy_source）──")
    for c in cases:
        d2 = next(x for x in m2["details"] if x["id"] == c["question"]["id"])
        d21 = next(x for x in m21["details"] if x["id"] == c["question"]["id"])
        obs = c["v21_obs"]
        flag = "✅" if d2["class"] == d21["class"] else "⚠️"
        print(
            f"    {c['question']['id']:>16} [{c['question']['type'][:12]:<12}] "
            f"v2={d2['class']:>6} v2.1={d21['class']:>6} {flag} "
            f"grader={obs['grader_called']} src={obs['policy_source']:<12} "
            f"route21={c['v21_route']}"
        )

    out = OUT_DIR / f"step14_cost_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "note": "Step 14 A/B: Frozen v2 vs Cost-aware v2.1 on frozen dev benchmark (18)",
                "capability": {"v2": cap2, "v2_1": cap21},
                "cost": {
                    "n": n,
                    "grader_calls_v2": n,
                    "grader_calls_v21": grader21,
                    "grader_reduction_pct": round(grader_reduction, 1),
                    "llm_calls_v2": llm_calls2,
                    "llm_calls_v21": llm_calls21,
                    "latency_v2": _stats(lat2),
                    "latency_v21": _stats(lat21),
                    "latency_reduction_pct": round(latency_reduction, 1),
                    "timeout_v21": op_err21,
                    "fallback_v21": fallback21,
                    "policy_llm_calls_v21": policy_llm21,
                    "retrieval_calls_v2": retr2,
                    "retrieval_calls_v21": retr21,
                    "grader_reasons": grader_reasons,
                },
                "details": [
                    {
                        "id": c["question"]["id"],
                        "type": c["question"]["type"],
                        "class_v2": next(x["class"] for x in m2["details"] if x["id"] == c["question"]["id"]),
                        "class_v21": next(x["class"] for x in m21["details"] if x["id"] == c["question"]["id"]),
                        "v21_obs": c["v21_obs"],
                        "route_v21": c["v21_route"],
                    }
                    for c in cases
                ],
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

"""
Step 13.5: Frozen Holdout Generalization Gate —— 一次性验收 v2（不可再调）

Holdout: tests/benchmark_holdout.json（16 题 / 7 类，开发期间未看 failure）
对比:   V0 Fixed Hybrid RAG  vs  Agentic v1  vs  Agentic v2（全部冻结）
要求:   run 前完成 ERROR≠ABSTAIN 语义修正（见 scripts/step14 相关），
        跑完不得再拿这 16 题调 v2 —— holdout 变 dev 就是评测纪律失效。

核心验证（Step 13 因果结论是否迁移）:
  - Final Rescue > 0 的前提：holdout 里存在 V0 miss（rescue market）
  - Harm = 0（安全门槛）
  - False Abstain = 0 或不再出现系统性误拒
  - OOD 继续正确拒答
  - Hop Recall / Completeness 明显优于 v1
  - Unnecessary Action 接近 0（不能为 Rescue 变成"什么都分解"）
  - Operational Failure Rate（timeout/model error）单独统计，不计入 OOD Reject / False Abstain

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step135_holdout_eval.py

产出: eval_results/step135_holdout_<timestamp>.json
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


def main():
    print("=" * 70)
    print("  🔒 Step 13.5: Frozen Holdout Generalization Gate（一次性验收）")
    print("=" * 70, flush=True)

    from eval.rescue_metrics import (
        compute_agent_capability_metrics,
        evidence_recall_at_k,
        hop_gold_ids,
    )
    from src.agentic_rag import AgenticRAG
    from src.agentic_rag_v1_backup import AgenticRAG as AgenticRAGv1
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
    agent_v1 = AgenticRAGv1(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    agent_v2 = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    bench = json.load(open("tests/benchmark_holdout.json", encoding="utf-8"))["benchmark"]
    print(f"  📝 Holdout: {len(bench)} 题（unseen，仅此一次验证）", flush=True)

    cases = []
    t0 = time.time()
    for i, b in enumerate(bench, 1):
        question = b["question"]
        qtype = b["type"]
        all_gold = set()
        for hg in hop_gold_ids(b):
            all_gold |= hg
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{qtype}] {question[:40]}", flush=True)

        # ── V0 Fixed RAG：单轮 hybrid → rerank → Top5（冻结 baseline）──
        v0_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:FETCH_K]
        v0_top5 = reranker.rerank(question, list(v0_cands), TOP_K)
        v0_er = evidence_recall_at_k(v0_top5, all_gold)

        # ── Agentic v1（冻结，从 v1 backup 导入）──
        r1 = agent_v1.run(question, fetch_k=FETCH_K, verbose=False)
        v1_top5 = r1["sources"][:TOP_K]
        v1_er = evidence_recall_at_k(v1_top5, all_gold)

        # ── Agentic v2（冻结候选）──
        r2 = agent_v2.run(question, fetch_k=FETCH_K, verbose=False)
        v2_top5 = r2["sources"][:TOP_K]
        v2_er = evidence_recall_at_k(v2_top5, all_gold)

        print(
            f"    V0_ER={v0_er:.2f} V1_ER={v1_er:.2f} V2_ER={v2_er:.2f} "
            f"route_v2={r2['route']} abstain={r2['abstained']} iters={r2['iterations']}",
            flush=True,
        )

        cases.append(
            {
                "question": b,
                "v0_sources": v0_top5,
                "v1_sources": v1_top5,
                "v2_sources": v2_top5,
                "v1_route": r1["route"],
                "v2_route": r2["route"],
                "v1_answer": r1["answer"],
                "v2_answer": r2["answer"],
                "v1_abstained": r1["abstained"],
                "v2_abstained": r2["abstained"],
                "v1_iterations": r1["iterations"],
                "v2_iterations": r2["iterations"],
                "v0_evidence_recall": v0_er,
            }
        )

    elapsed = time.time() - t0

    # ── 指标：v1 与 v2 分别相对 V0 计算 ──
    m1 = compute_agent_capability_metrics(
        [
            {
                "question": c["question"],
                "v0_sources": c["v0_sources"],
                "v1_sources": c["v1_sources"],
                "v1_route": c["v1_route"],
                "v1_answer": c["v1_answer"],
                "v1_abstained": c["v1_abstained"],
            }
            for c in cases
        ]
    )
    m2 = compute_agent_capability_metrics(
        [
            {
                "question": c["question"],
                "v0_sources": c["v0_sources"],
                "v1_sources": c["v2_sources"],
                "v1_route": c["v2_route"],
                "v1_answer": c["v2_answer"],
                "v1_abstained": c["v2_abstained"],
            }
            for c in cases
        ]
    )

    # ── 三版对比报告 ──
    def _fmt(m):
        return {
            "final_answer_accuracy": m["final_answer_accuracy"],
            "evidence_recall": m["evidence_recall"],
            "hop_recall": m["hop_recall"],
            "completeness": m["completeness"],
            "final_rescue": m["final_rescue"],
            "harm": m["harm"],
            "net_utility": m["net_utility"],
            "ood_reject": m["ood_reject"],
            "false_abstain": m["false_abstain"],
            "policy_action_accuracy": m["policy_action_accuracy"],
            "decomposition_success": m["decomposition_success"],
            "retry_recovery": m["retry_recovery"],
            "unnecessary_action_rate": m["unnecessary_action_rate"],
            "avg_iterations": m["avg_iterations"],
        }

    print("\n" + "=" * 70)
    print("  📊 Holdout 三版本对比（V0 vs v1 vs v2）")
    print("=" * 70)
    print(f"  {'指标':<26}{'V0':>10}{'v1':>10}{'v2':>10}")
    print(f"  {'-' * 58}")
    keys = [
        ("Final Answer Acc", "final_answer_accuracy"),
        ("Evidence Recall@5", "evidence_recall"),
        ("Hop Recall@5", "hop_recall"),
        ("Completeness", "completeness"),
        ("Final Rescue", "final_rescue"),
        ("Harm", "harm"),
        ("NetUtility", "net_utility"),
        ("OOD Reject", "ood_reject"),
        ("False Abstain", "false_abstain"),
        ("Policy Action Acc", "policy_action_accuracy"),
        ("Decomp Success", "decomposition_success"),
        ("Retry Recovery", "retry_recovery"),
        ("Unnecessary Action", "unnecessary_action_rate"),
        ("Avg Iterations", "avg_iterations"),
    ]
    v0_fh = sum(1 for c in cases if c["v0_evidence_recall"] > 0 and any(hop_gold_ids(c["question"])))
    for label, k in keys:
        if k == "Final Answer Acc" or k in ("OOD Reject", "False Abstain", "Policy Action Acc", "Unnecessary Action"):
            row = ["—", m1[k], m2[k]]
        elif k == "Avg Iterations":
            row = ["1", m1[k], m2[k]]
        elif k == "Final Rescue" or k == "Harm" or k == "NetUtility" or k == "Decomp Success" or k == "Retry Recovery":
            row = ["—", m1[k], m2[k]]
        else:
            row = [round(v0_fh / max(len(cases), 1), 3), m1[k], m2[k]]
        print(f"  {label:<26}{str(row[0]):>10}{str(row[1]):>10}{str(row[2]):>10}")

    # ── Rescue Market 报告（Step 13.5 要求先回答的问题）──
    n_ans = sum(1 for c in cases if c["question"]["type"] != "unsupported_ood")
    v0_misses = [c for c in cases if c["v0_evidence_recall"] == 0 and any(hop_gold_ids(c["question"]))]
    print("\n  ── Rescue Market（holdout 内 V0 miss）──")
    print(f"  Holdout Answerable      = {n_ans}")
    print(f"  V0 already-hit          = {n_ans - len(v0_misses)}")
    print(f"  V0 miss (Rescue market) = {len(v0_misses)}")
    for c in v0_misses:
        d1 = next(x for x in m1["details"] if x["id"] == c["question"]["id"])
        d2 = next(x for x in m2["details"] if x["id"] == c["question"]["id"])
        print(
            f"    {c['question']['id']:>16} [{c['question']['type']:>20}] "
            f"v1_ER={d1['v1_evidence_recall']} class={d1['class']:>6} | "
            f"v2_ER={d2['v1_evidence_recall']} class={d2['class']:>6}"
        )

    # ── Operational Failure Rate（单独统计，不计入 ABSTAIN）──
    op_fail = sum(1 for c in cases if "[LLM 不可用" in c["v2_answer"] or "系统错误" in c["v2_answer"])
    print("\n  ── Operational（不参与能力判定）──")
    print(f"  v2 Operational Failure = {op_fail}/{len(cases)}")
    for c in cases:
        if "[LLM 不可用" in c["v2_answer"] or "系统错误" in c["v2_answer"]:
            print(f"    ⚠️  {c['question']['id']}: {c['v2_answer'][:60]}")

    out = OUT_DIR / f"step135_holdout_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "note": "Step 13.5 one-shot holdout validation. Do NOT tune v2 on these 16 cases.",
                "v1": _fmt(m1),
                "v2": _fmt(m2),
                "rescue_market": {
                    "answerable": n_ans,
                    "v0_already_hit": n_ans - len(v0_misses),
                    "v0_miss": len(v0_misses),
                },
                "operational_failure_v2": op_fail,
                "details_v1": m1["details"],
                "details_v2": m2["details"],
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

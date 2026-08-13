"""
Step 16: LangGraph Runtime Parity Test —— Custom Runner vs LangGraph Adapter

研究问题：把 Agentic RAG v2 的 while-loop orchestration 套成 LangGraph
StateGraph runtime 后，行为是否完全一致？

同一组输入、同一 index、同一 prompt、同一 budget：
  Custom Runner（src/agentic_rag.py AgenticRAG.run）
  vs
  LangGraph Runner（src/langgraph_agent.py LangGraphAgenticRAG.run）

验证维度（final_step.md Step 16）：
  Answer / Evidence Recall
  Rescue / Harm / NetUtility
  OOD Reject / False Abstain
  Policy Route / Iterations

目标不是 LangGraph 更强，而是 **behavioral parity**。

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step16_runtime_parity.py

产出: eval_results/step16_runtime_parity_<timestamp>.json
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


def _route_equal(r1: list[str], r2: list[str]) -> bool:
    """route 完全一致（决策序列相同）"""
    return r1 == r2


def _answer_equal(a1: str, a2: str) -> bool:
    """回答一致：完全相同，或双方都是拒答（ABSTAIN 回答随 reason 文本可能略异）"""
    if a1 == a2:
        return True
    return "知识库中未找到" in a1 and "知识库中未找到" in a2


def main():
    # 分块运行（Windows pyarrow/milvus-lite 偶发段错误，分块可断点续跑）
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="起始题号（1-based）")
    parser.add_argument("--end", type=int, default=18, help="结束题号（含）")
    args = parser.parse_args()

    print("=" * 70)
    print("  🔀 Step 16: LangGraph Runtime Parity Test（Custom vs LangGraph）")
    print("=" * 70, flush=True)

    from eval.rescue_metrics import compute_agent_capability_metrics, evidence_recall_at_k, hop_gold_ids
    from src.agentic_rag import AgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.langgraph_agent import LangGraphAgenticRAG
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    # ⚠️ 初始化顺序：先 reranker 再 Milvus（Windows 上 Milvus 的 numpy memmap 会与
    # transformers 分片加载冲突导致段错误，见 step15 调试记录）
    reranker = CrossEncoderReranker()
    reranker._load_model()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
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

    # 同一实例：custom runner 与 LangGraph 包装器共享同一个 agent 实例，
    # 确保 LLM 调用、reranker、retriever 完全同源
    agent = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    lg_agent = LangGraphAgenticRAG(agent)

    bench = json.load(open("tests/benchmark_multi_hop.json", encoding="utf-8"))["benchmark"]
    print(f"  📝 Frozen Dev Benchmark: {len(bench)} 题", flush=True)

    cases = []
    t0 = time.time()
    n_route_diff = 0
    n_answer_diff = 0
    for i, b in enumerate(bench, 1):
        if i < args.start or i > args.end:
            continue
        question = b["question"]
        all_gold = set()
        for hg in hop_gold_ids(b):
            all_gold |= hg
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{b['type']}] {question[:40]}", flush=True)

        # ── Custom Runner ──
        r_custom = agent.run(question, fetch_k=FETCH_K, verbose=False)
        custom_top5 = r_custom["sources"][:TOP_K]
        custom_er = evidence_recall_at_k(custom_top5, all_gold)
        custom_top5 = r_custom["sources"][:TOP_K]
        custom_er = evidence_recall_at_k(custom_top5, all_gold)

        # ── LangGraph Runner ──
        r_lg = lg_agent.run(question, fetch_k=FETCH_K, verbose=False)
        lg_top5 = r_lg["sources"][:TOP_K]
        lg_er = evidence_recall_at_k(lg_top5, all_gold)

        route_same = _route_equal(r_custom["route"], r_lg["route"])
        answer_same = _answer_equal(r_custom["answer"], r_lg["answer"])
        abstain_same = r_custom["abstained"] == r_lg["abstained"]
        er_same = custom_er == lg_er
        iters_same = r_custom["iterations"] == r_lg["iterations"]

        n_route_diff += not route_same
        n_answer_diff += not answer_same
        flag = "✅" if (route_same and answer_same and abstain_same and er_same and iters_same) else "⚠️"
        print(
            f"    route={'同' if route_same else '异'} answer={'同' if answer_same else '异'} "
            f"abstain={'同' if abstain_same else '异'} ER={'同' if er_same else '异'} iters={'同' if iters_same else '异'} {flag}",
            flush=True,
        )
        if not route_same:
            print(f"      custom: {r_custom['route']}")
            print(f"      lg    : {r_lg['route']}")

        cases.append(
            {
                "question": b,
                "custom_route": r_custom["route"],
                "lg_route": r_lg["route"],
                "custom_answer": r_custom["answer"],
                "lg_answer": r_lg["answer"],
                "custom_abstained": r_custom["abstained"],
                "lg_abstained": r_lg["abstained"],
                "custom_iterations": r_custom["iterations"],
                "lg_iterations": r_lg["iterations"],
                "custom_sources": custom_top5,
                "lg_sources": lg_top5,
                "custom_er": custom_er,
                "lg_er": lg_er,
                "route_equal": route_same,
                "answer_equal": answer_same,
                "er_equal": er_same,
            }
        )

    elapsed = time.time() - t0

    # ── 汇总 ──
    def _case(suffix: str) -> list[dict]:
        return [
            {
                "question": c["question"],
                "v0_sources": c[f"{suffix}_sources"],
                "v1_sources": c[f"{suffix}_sources"],
                "v1_route": c[f"{suffix}_route"],
                "v1_answer": c[f"{suffix}_answer"],
                "v1_abstained": c[f"{suffix}_abstained"],
            }
            for c in cases
        ]

    n = len(cases)
    m_custom = compute_agent_capability_metrics(_case("custom"))
    m_lg = compute_agent_capability_metrics(_case("lg"))

    parity = {
        "route_exact_match": f"{n - n_route_diff}/{n}",
        "answer_match": f"{n - n_answer_diff}/{n}",
        "abstain_match": sum(1 for c in cases if c["custom_abstained"] == c["lg_abstained"]),
        "er_match": sum(1 for c in cases if c["er_equal"]),
        "iterations_match": sum(1 for c in cases if c["custom_iterations"] == c["lg_iterations"]),
    }

    print("\n" + "=" * 70)
    print("  🔀 Parity 汇总（Custom vs LangGraph）")
    print("=" * 70)
    for k, v in parity.items():
        print(f"  {k:<20}= {v}")

    print(f"\n  {'能力维度':<22}{'Custom':>10}{'LangGraph':>10}")
    print(f"  {'-' * 44}")
    for k in ["final_answer_accuracy", "evidence_recall", "hop_recall", "completeness", "final_rescue", "harm"]:
        print(f"  {k:<22}{str(m_custom[k]):>10}{str(m_lg[k]):>10}")
    print(f"  {'ood_reject':<22}{str(m_custom['ood_reject']):>10}{str(m_lg['ood_reject']):>10}")
    print(f"  {'false_abstain':<22}{str(m_custom['false_abstain']):>10}{str(m_lg['false_abstain']):>10}")
    print(
        f"  {'policy_action_acc':<22}{str(m_custom['policy_action_accuracy']):>10}{str(m_lg['policy_action_accuracy']):>10}"
    )

    all_match = n_route_diff == 0 and n_answer_diff == 0 and parity["er_match"] == n
    print(f"\n  {'✅ PARITY PASS' if all_match else '⚠️ PARITY MISMATCH'}（route/answer/ER 全一致 = pass）")

    out = OUT_DIR / f"step16_runtime_parity_{TIMESTAMP}_b{args.start}-{args.end}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "note": "Step 16: LangGraph Runtime Parity（Custom vs LangGraph，同一 input/index/prompt/budget）",
                "parity": parity,
                "all_match": all_match,
                "capability_custom": {k: v for k, v in m_custom.items() if k != "details"},
                "capability_langgraph": {k: v for k, v in m_lg.items() if k != "details"},
                "details": [
                    {
                        "id": c["question"]["id"],
                        "type": c["question"]["type"],
                        "route_equal": c["route_equal"],
                        "answer_equal": c["answer_equal"],
                        "er_equal": c["er_equal"],
                        "custom_route": c["custom_route"],
                        "lg_route": c["lg_route"],
                        "custom_er": c["custom_er"],
                        "lg_er": c["lg_er"],
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

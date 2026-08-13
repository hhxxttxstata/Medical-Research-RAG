"""
Step 10: Agentic RAG v1 评测脚本

对比两个系统（同一 81 题数据集）：
  V0 Baseline     : hybrid retrieval (fetch_k=20) → rerank(q_original) → Top5
                    （与 Step 1–9 冻结的 baseline comparator 一致）
  Agentic RAG v1  : Retrieve → Grade → [Accept / Retrieve / Decompose / Abstain]
                    max_iterations=2

指标（eval/rescue_metrics.py 统一口径）：
  - FinalHit@5（chunk-level gold，Step 8 标注）
  - Rescue / Harm / NetUtility（相对 V0 baseline）
  - ABSTAIN 正确率：OOD 问题是否正确拒答；answerable 问题是否错误拒答

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step10_agentic_eval.py

产出: eval_results/step10_agentic_eval_<timestamp>.json
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

from src.agentic_rag import AgenticRAG  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 5
FETCH_K = 20

from eval.rescue_metrics import gold_chunk_ids, hit_at_k  # noqa: E402


def main():
    print("=" * 70)
    print("  🔬 Step 10: Agentic RAG v1 vs V0 Baseline 评测")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    reranker = CrossEncoderReranker()
    reranker._load_model()
    print(f"  ✅ Reranker ready={reranker.model_ready}", flush=True)

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
    agent = AgenticRAG(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        max_iterations=2,
    )

    questions = json.load(open("tests/test_questions.json", encoding="utf-8"))
    print(f"  📝 问题集: {len(questions)}", flush=True)

    cases = []
    stats = {
        "v0_hit": 0,
        "agent_hit": 0,
        "rescue": 0,
        "harm": 0,
        "agent_abstain": 0,
        "agent_accept": 0,
        "ood_correct_abstain": 0,
        "ood_total": 0,
        "answerable_wrong_abstain": 0,
        "answerable_total": 0,
        "route_counts": {},
        "iterations_sum": 0,
    }

    t0 = time.time()
    for i, q in enumerate(questions, 1):
        question = q["question"]
        gold_ids = gold_chunk_ids(q)
        expected_doc = q.get("expected_doc", "")
        cat = q.get("category", "")
        is_ood = cat == "out_of_knowledge"

        print(f"  ── [{i}/{len(questions)}] {q['id']} {question[:40]}", flush=True)

        # ── V0 Baseline：hybrid Top20 → rerank(q_original) → Top5 ──
        v0_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:20]
        v0_top5 = reranker.rerank(question, list(v0_cands), TOP_K)
        v0_hit = hit_at_k(v0_top5, gold_ids, expected_doc, k=TOP_K)
        stats["v0_hit"] += v0_hit

        # ── Agentic RAG v1 ──
        agent_result = agent.run(question, fetch_k=FETCH_K, verbose=False)
        agent_sources = agent_result["sources"]
        agent_hit = hit_at_k(agent_sources, gold_ids, expected_doc, k=TOP_K)
        agent_abstained = agent_result["abstained"]

        stats["agent_hit"] += agent_hit
        stats["agent_abstain"] += agent_abstained
        stats["agent_accept"] += not agent_abstained
        stats["iterations_sum"] += agent_result["iterations"]
        route_key = "→".join(agent_result["route"])
        stats["route_counts"][route_key] = stats["route_counts"].get(route_key, 0) + 1

        # Rescue/Harm（相对 V0）
        if not v0_hit and agent_hit:
            stats["rescue"] += 1
        if v0_hit and not agent_hit and not agent_abstained:
            stats["harm"] += 1
        if v0_hit and agent_abstained:
            stats["harm"] += 1  # baseline 命中但 agent 拒答 = Harm

        # 拒答质量
        if is_ood:
            stats["ood_total"] += 1
            if agent_abstained:
                stats["ood_correct_abstain"] += 1
        elif q.get("expected_doc"):
            stats["answerable_total"] += 1
            if agent_abstained:
                stats["answerable_wrong_abstain"] += 1

        cases.append(
            {
                "id": q["id"],
                "question": question,
                "category": cat,
                "v0_hit": v0_hit,
                "agent_hit": agent_hit,
                "agent_abstained": agent_abstained,
                "agent_route": agent_result["route"],
                "agent_iterations": agent_result["iterations"],
                "agent_evidence_score": round(agent_result["state"].evidence_score, 3),
                "agent_decision": agent_result["state"].decision,
            }
        )

    elapsed = time.time() - t0

    # ── 汇总 ──
    n_labeled = len(questions)
    print("\n" + "=" * 70)
    print("  📊 Agentic RAG v1 vs V0 Baseline")
    print("=" * 70)
    print(f"  N = {n_labeled} 题（全部 81 题）")
    print(f"  V0 FinalHit@5     = {stats['v0_hit']}  ({stats['v0_hit'] / n_labeled:.0%})")
    print(f"  Agent FinalHit@5  = {stats['agent_hit']}  ({stats['agent_hit'] / n_labeled:.0%})")
    print(f"  🆘 Rescue         = {stats['rescue']}")
    print(f"  💥 Harm           = {stats['harm']}")
    print(f"  📈 NetUtility     = {stats['rescue'] - stats['harm']:+d}")
    print(f"  🚫 Agent ABSTAIN  = {stats['agent_abstain']} 题（route 分布: {stats['route_counts']}）")
    print(f"  ⏱  平均迭代数    = {stats['iterations_sum'] / n_labeled:.2f}")
    print("\n  🚫 拒答质量")
    print(
        f"  OOD 正确拒答      = {stats['ood_correct_abstain']}/{stats['ood_total']}  ({stats['ood_correct_abstain'] / max(stats['ood_total'], 1):.0%})"
    )
    print(f"  answerable 误拒   = {stats['answerable_wrong_abstain']}/{stats['answerable_total']}")

    out = OUT_DIR / f"step10_agentic_eval_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "stats": stats,
                "cases": cases,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

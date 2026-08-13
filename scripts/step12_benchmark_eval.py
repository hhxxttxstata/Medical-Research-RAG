"""
Step 12: Agentic Capability / Multi-hop Benchmark 评测

对比 V0 Fixed RAG vs Agentic RAG v1（冻结 baseline）on benchmark_multi_hop.json：
  - Final Answer Accuracy / Evidence Recall@K / Hop Recall@K / Completeness
  - Final Rescue / Harm / NetUtility（相对 V0）
  - OOD Reject / False Abstain
  - Policy Action Accuracy / Decomposition Success / Retry Recovery
  - Unnecessary Action Rate / Avg Iterations / Retrieval Calls / LLM Calls

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step12_benchmark_eval.py

产出: eval_results/step12_benchmark_<timestamp>.json
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
    print("  🔬 Step 12: Multi-hop / Agent Capability Benchmark")
    print("=" * 70, flush=True)

    from src.agentic_rag import AgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.lucene_bm25 import LuceneBM25Index
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
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
    agent = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    from eval.rescue_metrics import (
        compute_agent_capability_metrics,
        evidence_recall_at_k,
        hop_gold_ids,
    )

    bench = json.load(open("tests/benchmark_multi_hop.json", encoding="utf-8"))["benchmark"]
    print(f"  📝 Benchmark: {len(bench)} 题", flush=True)

    cases = []
    t0 = time.time()
    for i, b in enumerate(bench, 1):
        question = b["question"]
        qtype = b["type"]
        all_gold = set()
        for hg in hop_gold_ids(b):
            all_gold |= hg
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{qtype}] {question[:40]}", flush=True)

        # ── V0 Fixed RAG：单轮 hybrid → rerank → Top5 ──
        v0_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:FETCH_K]
        v0_top5 = reranker.rerank(question, list(v0_cands), TOP_K)
        v0_er = evidence_recall_at_k(v0_top5, all_gold)

        # ── Agentic v1 ──
        r = agent.run(question, fetch_k=FETCH_K, verbose=False)
        v1_top5 = r["sources"][:TOP_K]
        v1_er = evidence_recall_at_k(v1_top5, all_gold)

        print(
            f"    V0_ER={v0_er:.2f} Agent_ER={v1_er:.2f} route={r['route']} "
            f"abstain={r['abstained']} iters={r['iterations']}",
            flush=True,
        )

        cases.append(
            {
                "question": b,
                "v0_sources": v0_top5,
                "v1_sources": v1_top5,
                "v1_route": r["route"],
                "v1_answer": r["answer"],
                "v1_abstained": r["abstained"],
                "v1_iterations": r["iterations"],
                "v1_llm_calls": r["state"].iteration * 2 if r["state"] else 0,
                "v0_evidence_recall": v0_er,
            }
        )

    elapsed = time.time() - t0

    # ── 汇总 ──
    metrics = compute_agent_capability_metrics(cases)
    print("\n" + "=" * 70)
    print("  📊 Step 12 Benchmark 结果（V0 vs Agentic v1）")
    print("=" * 70)
    print(f"  N = {metrics['n']} 题")
    print(f"  Final Answer Accuracy  = {metrics['final_answer_accuracy']}")
    print(f"  Evidence Recall@5      = {metrics['evidence_recall']}")
    print(f"  Hop Recall@5           = {metrics['hop_recall']}")
    print(f"  Completeness           = {metrics['completeness']}")
    print(f"  🆘 Final Rescue        = {metrics['final_rescue']}")
    print(f"  💥 Harm                = {metrics['harm']}")
    print(f"  📈 NetUtility          = {metrics['net_utility']}")
    print(f"  🚫 OOD Reject          = {metrics['ood_reject']}")
    print(f"  ⚠️  False Abstain       = {metrics['false_abstain']}")
    print(f"  🎯 Policy Action Acc    = {metrics['policy_action_accuracy']}")
    print(f"  🔗 Decomp Success       = {metrics['decomposition_success']}")
    print(f"  🔄 Retry Recovery       = {metrics['retry_recovery']}")
    print(f"  🚫 Unnecessary Actions  = {metrics['unnecessary_action_rate']}")
    print(f"  ⏱  Avg Iterations       = {metrics['avg_iterations']}")
    print("\n  ── 分类型 ──")
    for t, s in metrics["by_type"].items():
        print(f"    [{t:>22}] n={s['n']} hit={s['hit']} rescue={s['rescue']} false_abstain={s['false_abstain']}")

    # ── Failure Anatomy：V0 miss 的题 ──
    print("\n  ── Failure Anatomy（V0 Evidence Recall=0 的题）──")
    for d in metrics["details"]:
        if d["v0_evidence_recall"] == 0 and d["type"] != "unsupported_ood":
            print(
                f"    {d['id']:>16} [{d['type']:>22}] V0_ER=0 Agent_ER={d['v1_evidence_recall']} "
                f"route={d['route']} rescue={'✅' if d['class'] == 'rescue' else '❌'}"
            )

    out = OUT_DIR / f"step12_benchmark_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "metrics": {k: v for k, v in metrics.items() if k != "details"},
                "details": metrics["details"],
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

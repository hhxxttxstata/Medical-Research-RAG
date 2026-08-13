"""
Step 15: End-to-End Grounded Answer Evaluation（frozen evaluation，最后 1 次质量实验）

在冻结的 dev benchmark（18 题）+ frozen Agentic RAG v2 上运行，评测最终生成答案
的 grounded 质量——补齐 RAG lifecycle：retrieval → answer → claim-level grounding。

能力证据缺口：之前所有评测都止步于 "Evidence Recall / Final Answer Accuracy"，
没有回答"最终答案中的每个 factual claim 是否由 final_evidence 支撑"。

6 项核心指标 + Correct Abstention：
  Answer Correctness / Groundedness / Evidence-Citation Support /
  Completeness / Unsupported Claim Rate / Correct Abstention

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step15_grounded_eval.py

产出: eval_results/step15_grounded_<timestamp>.json
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
    # 分块运行（Windows pyarrow/milvus-lite 偶发段错误，分块可断点续跑）
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="起始题号（1-based）")
    parser.add_argument("--end", type=int, default=18, help="结束题号（含）")
    args = parser.parse_args()

    print("=" * 70)
    print("  📝 Step 15: End-to-End Grounded Answer Evaluation（frozen）")
    print("=" * 70, flush=True)

    from eval.grounded_metrics import compute_grounded_metrics
    from src.agentic_rag import AgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
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
    agent = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    bench = json.load(open("tests/benchmark_multi_hop.json", encoding="utf-8"))["benchmark"]
    print(f"  📝 Frozen Dev Benchmark: {len(bench)} 题（Step 12/13 同款，冻结）", flush=True)

    cases = []
    t0 = time.time()
    for i, b in enumerate(bench, 1):
        if i < args.start or i > args.end:
            continue
        question = b["question"]
        qtype = b["type"]
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{qtype}] {question[:40]}", flush=True)

        # ── Frozen Agentic RAG v2（同一 pipeline，无任何新能力）──
        r = agent.run(question, fetch_k=FETCH_K, verbose=False)
        top5 = r["sources"][:TOP_K]
        answer = r["answer"]
        op_err = answer.startswith("[OPERATIONAL_ERROR]")

        cases.append(
            {
                "question": b,
                "answer": answer,
                "sources": top5,
                "abstained": r["abstained"],
                "operational_error": op_err,
                "route": r["route"],
                "elapsed": r["elapsed"],
            }
        )
        print(f"    route={r['route']} abstain={r['abstained']} op_err={op_err}", flush=True)

    elapsed = time.time() - t0

    # ── Grounded 指标（LLM-as-Judge，claim 级）──
    metrics = compute_grounded_metrics(cases, generator=generator)
    print("\n" + "=" * 70)
    print("  📊 Step 15 Grounded Answer Metrics（frozen v2 on dev benchmark）")
    print("=" * 70)
    for k in [
        "Answer Correctness",
        "Groundedness",
        "Evidence/Citation Support",
        "Completeness",
        "Unsupported Claim Rate",
        "Correct Abstention",
        "False Abstain",
        "Citation Valid Rate",
    ]:
        print(f"  {k:<28}= {metrics[k]}")
    print(f"  claim 总数        = {metrics['claim_total']}（unsupported {metrics['unsupported_claim_total']}）")
    print(f"  Operational Error = {sum(1 for c in cases if c['operational_error'])}/{len(cases)}")

    # ── Unsupported Claim 明细（Failure Anatomy，面试可讲）──
    print("\n  ── Unsupported Claims（每条显式记录）──")
    if metrics["unsupported_claims"]:
        for uc in metrics["unsupported_claims"]:
            print(f"    {uc['id']:>16}: {uc['claim'][:60]}")
    else:
        print("    （无）")

    # ── 逐题 ──
    print("\n  ── 逐题（grounded / unsupported / correctness）──")
    for det in metrics["details"]:
        if det["abstained"]:
            line = f"    {det['id']:>16} [ABSTAIN]"
        else:
            g = sum(1 for c in det["claims"] if c["status"] == "supported")
            u = sum(1 for c in det["claims"] if c["status"] == "unsupported")
            f = sum(1 for c in det["claims"] if c["status"] != "unverifiable")
            line = f"    {det['id']:>16} grounded={g}/{f} unsup={u} correct={det['answer_correct']} [{det['mode']}]"
        print(line)

    out = OUT_DIR / f"step15_grounded_{TIMESTAMP}_b{args.start}-{args.end}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "note": "Step 15: End-to-End Grounded Answer Eval on frozen dev benchmark (18), Agentic RAG v2 frozen",
                "metrics": {k: v for k, v in metrics.items() if k != "details"},
                "details": metrics["details"],
                "operational_error": sum(1 for c in cases if c["operational_error"]),
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

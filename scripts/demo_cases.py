"""
Demo Cases —— 4 个精选演示（面试展示素材）

覆盖 Agentic RAG 的 4 种决策路径，每个 case 展示 route + 回答 + 成本观测：

  1. bh_easy_01  easy_single_hop     → ACCEPT（cheap signal，零 grader）
  2. bh_multi_01 multi_hop_composite → DECOMPOSE（hop plan 执行）
  3. bh_comp_01  comparison          → DECOMPOSE（对比拆解）
  4. bh_ood_02   unsupported_ood     → ABSTAIN（时间敏感 → grader 裁决拒答）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/demo_cases.py

产出: eval_results/demo_cases_<timestamp>.json
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

# 4 个演示 case（id 取自冻结 dev benchmark，覆盖 4 种决策路径）
DEMO_IDS = ["bh_easy_01", "bh_multi_01", "bh_comp_01", "bh_ood_02"]


def main():
    print("=" * 70)
    print("  🎬 Demo Cases —— Agentic RAG v2.1 演示（4 种决策路径）")
    print("=" * 70, flush=True)

    from src.cost_aware_agentic_rag import CostAwareAgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    # ⚠️ 初始化顺序：先 reranker 再 Milvus（Windows 段错误规避）
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
    agent = CostAwareAgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    bench = {b["id"]: b for b in json.load(open("tests/benchmark_multi_hop.json", encoding="utf-8"))["benchmark"]}

    results = []
    t0 = time.time()
    for qid in DEMO_IDS:
        b = bench[qid]
        question = b["question"]
        print(f"\n  ── {qid} [{b['type']}] {question}", flush=True)

        c0 = generator.call_counts()
        r = agent.run(question, fetch_k=FETCH_K, verbose=False)
        c1 = generator.call_counts()
        llm_calls = {k: c1[k] - c0[k] for k in c1}
        obs = r["observation"]

        results.append(
            {
                "id": qid,
                "type": b["type"],
                "question": question,
                "route": r["route"],
                "abstained": r["abstained"],
                "answer": r["answer"][:800],
                "observation": obs,
                "llm_calls": llm_calls,
                "elapsed": r["elapsed"],
            }
        )

        print(f"    route = {r['route']}")
        print(f"    grader_called = {obs['grader_called']}  policy_source = {obs['policy_source']}")
        print(f"    LLM calls = {llm_calls['total']}  elapsed = {r['elapsed']}s")
        if r["abstained"]:
            print(f"    answer = [拒答] {r['answer'][:80]}...")
        else:
            print(f"    answer = {r['answer'][:120]}...")

    elapsed = time.time() - t0
    out = OUT_DIR / f"demo_cases_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "note": "Demo Cases: 4 种决策路径",
                "cases": results,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 演示输出: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

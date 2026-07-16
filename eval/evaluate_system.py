"""
RAG 系统性能评估 — 直接测检索质量，不依赖 LLM API
用法: python -m eval.evaluate_system
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.metrics import compute_all_metrics, print_metrics_report
from eval.test_questions import get_test_questions
from src.rag_pipeline import RAGPipeline


def main():
    questions = get_test_questions()
    exact_qs = [q for q in questions if q.get("category") == "exact_match"]
    print(f"\n📋 测试: {len(questions)} 题 (精确匹配: {len(exact_qs)} 题)\n")

    pipeline = RAGPipeline(
        data_dir="data",
        top_k=10,
        enable_rewrite=False,
        enable_reranker=False,
        milvus_lite=True,
        vector_backend="milvus",
    )

    count = pipeline.initialize_knowledge_base(force_reindex=True)
    if count == 0:
        print("❌ 知识库为空，退出")
        return
    print(f"\n📚 知识库: {count} chunks\n")

    # warmup embedding model
    if hasattr(pipeline.embedding_provider, "warmup"):
        pipeline.embedding_provider.warmup()

    # ensure BM25 index
    pipeline.retriever._ensure_bm25_index()

    records = []
    for q in questions:
        question = q["question"]
        cat = q.get("category", "unknown")
        expected = q.get("expected_doc", "")
        diff = q.get("difficulty", "unknown")

        t0 = time.time()

        # vector search
        emb = pipeline.embedding_provider.embed([question], prefix="query: ")[0]
        vec_results = pipeline.vector_store.similarity_search(query_embedding=emb, top_k=20)

        # BM25
        bm25_results = pipeline.retriever._bm25_retrieve(question, top_k=20)

        # RRF fusion
        if bm25_results and vec_results:
            sources = pipeline.retriever._rrf_fusion(vec_results, bm25_results, top_k=10)
        elif vec_results:
            sources = vec_results[:10]
        else:
            sources = []

        elapsed = time.time() - t0
        vec_score = sources[0].get("_vector_score", 0) if sources else 0
        rrf_score = sources[0]["score"] if sources else 0
        top_score = vec_score or rrf_score

        expected_hit = False
        if expected and sources:
            expected_base = expected.rsplit(".", 1)[0]
            expected_hit = any(
                expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
                for s in sources
            )

        records.append(
            {
                "question": question,
                "category": cat,
                "difficulty": diff,
                "expected_doc": expected,
                "expected_hit": expected_hit,
                "num_retrieved": len(sources),
                "top_score": round(top_score, 4),
                "time_seconds": round(elapsed, 2),
                "sources": sources,
            }
        )

        status = "✅" if expected_hit else (" - " if not expected else "❌")
        print(f"  {status} [{cat[:4]}/{diff[:4]}] {question[:45]:<45s} score={top_score:.3f} time={elapsed:.2f}s")

    metrics = compute_all_metrics(records)
    print_metrics_report(metrics)

    out = os.path.join(os.path.abspath("eval_results"), f"system_eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "records": records}, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 报告: {out}")

    pipeline.close()


if __name__ == "__main__":
    main()

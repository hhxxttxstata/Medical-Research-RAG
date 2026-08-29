"""
Ragas 正式评测 — 与现有评测互为交叉验证

用法:
    python eval/run_ragas.py

依赖:
    pip install -r requirements.txt -r requirements-eval.txt
    （ragas/datasets 不在主清单，CI 不安装）
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset

from src.rag_pipeline import RAGPipeline

SKIP_REINDEX = "--skip-reindex" in sys.argv


def _build_eval_dataset() -> Dataset:
    """加载测试题 → 逐题检索并存 retrieval 结果"""
    from eval.test_questions import get_test_questions

    questions = get_test_questions()

    pipeline = RAGPipeline(
        data_dir="data",
        top_k=10,
        enable_rewrite=False,
        enable_reranker=False,
        milvus_lite=True,
        vector_backend="milvus",
    )

    if not SKIP_REINDEX:
        pipeline.initialize_knowledge_base(force_reindex=True)

    pipeline.retriever._ensure_bm25_index()

    rows = []
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")
        ground_truth = q.get("answer", "")

        # 混合检索
        emb = pipeline.embedding_provider.embed([question], prefix="query: ")[0]
        vec_results = pipeline.vector_store.similarity_search(query_embedding=emb, top_k=20)
        bm25_results = pipeline.retriever._bm25_retrieve(question, top_k=20)

        if bm25_results and vec_results:
            sources = pipeline.retriever._rrf_fusion(vec_results, bm25_results, top_k=10)
        elif vec_results:
            sources = vec_results[:10]
        else:
            sources = []

        contexts = [s["text"] for s in sources]

        # 生成回答（只跑前 40 题，Ragas 评测 LLM 调用量大）
        answer = ""
        if hasattr(pipeline.generator, "chat") and len(rows) < 40:
            try:
                from src.generator import build_rag_prompt

                prompt_data = build_rag_prompt(question, sources)
                gen = pipeline.generator.generate_structured(prompt_data, self_reflect=False)
                answer = gen.get("raw", "")
            except Exception:
                answer = "(generation skipped)"

        rows.append(
            {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth,
            }
        )

    pipeline.close()
    return Dataset.from_list(rows)


def main():
    print("=" * 60)
    print("  Ragas 正式评测")
    print("=" * 60)

    t0 = time.time()
    dataset = _build_eval_dataset()
    print(f"  ✅ 构建数据集: {len(dataset)} 条 ({time.time() - t0:.1f}s)\n")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        print("  📊 运行 Ragas evaluate()...")
        t1 = time.time()

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        elapsed = time.time() - t1
        print(f"\n  ⏱  评测耗时: {elapsed:.1f}s\n")

        report = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "num_samples": len(dataset),
            "metrics": {
                "faithfulness": round(result.get("faithfulness", 0), 4),
                "answer_relevancy": round(result.get("answer_relevancy", 0), 4),
                "context_precision": round(result.get("context_precision", 0), 4),
                "context_recall": round(result.get("context_recall", 0), 4),
            },
        }

        print("  📊 Ragas 指标")
        print(f"     Faithfulness:      {report['metrics']['faithfulness']:.4f}")
        print(f"     Answer Relevancy:  {report['metrics']['answer_relevancy']:.4f}")
        print(f"     Context Precision: {report['metrics']['context_precision']:.4f}")
        print(f"     Context Recall:    {report['metrics']['context_recall']:.4f}")

        # 保存
        out_dir = Path("eval_results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"ragas_eval_{report['timestamp']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 报告: {out_path}")

    except ImportError as e:
        print(f"  ⚠️  Ragas 未安装或依赖缺失: {e}")
        print("  💡  pip install ragas datasets pandas")
        sys.exit(1)


if __name__ == "__main__":
    main()

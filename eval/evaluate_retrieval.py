"""
检索组件 A/B 对比评估

独立于生成环节，只评估检索阶段的质量。
支持对比：
  - 不同 embedding 模型
  - Query Rewriting 开/关
  - Reranker 开/关
  - 纯向量检索 vs 混合检索（向量 + BM25）

使用方式：
    python -m eval.evaluate_retrieval
    python -m eval.evaluate_retrieval --ablation rewrite
    python -m eval.evaluate_retrieval --compare-embeddings
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import get_embedding_provider
from src.rag_pipeline import RAGPipeline

from .metrics import (
    compute_all_metrics,
)
from .test_questions import get_test_questions

# ── Embedding 模型列表 ──────────────────────────

EMBEDDING_MODELS = [
    {
        "name": "multilingual-e5-small",
        "provider": "local",
        "model": "intfloat/multilingual-e5-small",
        "dim": 384,
        "note": "当前默认（轻量，中英均可）",
    },
    {
        "name": "multilingual-e5-base",
        "provider": "local",
        "model": "intfloat/multilingual-e5-base",
        "dim": 768,
        "note": "e5 中等版本，质量更高",
    },
    {
        "name": "bge-m3",
        "provider": "local",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "note": "BGE 多语言，支持稠密+稀疏混合",
    },
]


def _calc_recall_at_k(expected_doc: str, sources: list[dict[str, Any]], k: int) -> int:
    """计算 Recall@K: 预期文档是否出现在 top-K 结果中"""
    for s in sources[:k]:
        fn = s.get("metadata", {}).get("filename", "")
        if expected_doc in fn:
            return 1
    return 0


def _calc_precision_at_k(sources: list[dict[str, Any]], k: int) -> float:
    """计算 Precision@K: 检索结果中有多少来自不同源文档"""
    # 简化：假设每个不同文件名算一个"相关"结果
    if not sources or k == 0:
        return 0.0
    top = sources[:k]
    filenames = set(s.get("metadata", {}).get("filename", "") for s in top if s.get("metadata"))
    return len(filenames) / min(k, len(top))


class RetrievalEvaluator:
    """检索组件 A/B 对比评估器"""

    def __init__(self, data_dir: str = "data", persist_dir: str = "chroma_db_eval"):
        self.data_dir = os.path.abspath(data_dir)
        self.persist_dir = os.path.abspath(persist_dir)
        self._clean_chroma()

    def _clean_chroma(self):
        """清空评估用向量数据库"""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

    def _build_pipeline(self, **kwargs) -> RAGPipeline:
        """构建标准 pipeline"""
        self._clean_chroma()
        pipeline = RAGPipeline(
            data_dir=self.data_dir,
            persist_dir=self.persist_dir,
            **kwargs,
        )
        pipeline.initialize_knowledge_base(force_reindex=True)
        return pipeline

    def run_ablation(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """消融实验：依次关闭 rewrite、reranker、hybrid，对比检索效果

        Returns:
            [{"variant": "full", "metrics": {...}}, ...]
        """
        exact_questions = [q for q in questions if q.get("category") == "exact_match"]
        results = []

        variants = [
            {
                "name": "full (rewrite+hybrid+rerank)",
                "retriever_mode": "hybrid",
                "enable_rewrite": True,
                "enable_reranker": True,
            },
            {
                "name": "no_rewrite",
                "retriever_mode": "hybrid",
                "enable_rewrite": False,
                "enable_reranker": True,
            },
            {
                "name": "no_reranker",
                "retriever_mode": "hybrid",
                "enable_rewrite": True,
                "enable_reranker": False,
            },
            {
                "name": "vector_only",
                "retriever_mode": "vector",
                "enable_rewrite": False,
                "enable_reranker": False,
            },
        ]

        for variant in variants:
            print(f"\n{'=' * 60}")
            print(f"  🔬 变体: {variant['name']}")
            print(f"{'=' * 60}")

            pipeline = self._build_pipeline(
                retriever_mode=variant["retriever_mode"],
                enable_rewrite=variant["enable_rewrite"],
                enable_reranker=variant["enable_reranker"],
            )

            variant_records = []
            for q in exact_questions:
                start = time.time()
                result = pipeline.query(q["question"], top_k=3)
                elapsed = time.time() - start

                sources = result.get("sources", [])
                expected_doc = q.get("expected_doc", "")
                expected_hit = any(expected_doc in s.get("metadata", {}).get("filename", "") for s in sources)

                variant_records.append(
                    {
                        "question": q["question"],
                        "expected_doc": expected_doc,
                        "expected_hit": expected_hit,
                        "num_retrieved": len(sources),
                        "top_score": sources[0]["score"] if sources else 0,
                        "time_seconds": round(elapsed, 2),
                        "category": q["category"],
                        "difficulty": q.get("difficulty", "unknown"),
                        "sources": sources,
                    }
                )

            variant_metrics = compute_all_metrics(variant_records)
            results.append(
                {
                    "variant": variant["name"],
                    "metrics": variant_metrics,
                    "records": variant_records,
                }
            )

        return results

    def run_embedding_comparison(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对比不同 embedding 模型的检索效果

        注意：每个模型需要重建向量库，全量比较耗时。
        这里采用快捷方式：使用默认 pipeline 检索，然后分别用不同
        embedding 模型计算 query 和 chunks 的语义相似度作为参考分。
        """
        print(f"\n{'=' * 60}")
        print("  🔬 开始 Embedding 模型对比")
        print(f"{'=' * 60}")

        # 先用默认模型建库
        pipeline = self._build_pipeline()
        all_questions = [q for q in questions if q.get("category") != "out_of_knowledge"]

        # 收集所有 chunk 文本
        chunks = pipeline.vector_store.get_all_documents()
        chunk_texts = [c["text"] for c in chunks]

        results = []
        for emb_config in EMBEDDING_MODELS:
            print(f"\n  评估模型: {emb_config['name']} ({emb_config['note']})")
            try:
                provider = get_embedding_provider(emb_config["provider"], emb_config["model"])
                provider.warmup()
            except Exception as e:
                print(f"    ⚠️ 模型加载失败: {e}")
                continue

            query_scores = []
            for q in all_questions:
                try:
                    q_emb = provider.embed([q["question"]], prefix="query: ")[0]
                except Exception:
                    continue

                # 对所有 chunk 计算余弦相似度，取 top-k
                chunk_scores = []
                # 分批处理以避免 OOM
                batch_size = 128
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i : i + batch_size]
                    try:
                        batch_embs = provider.embed(batch, prefix="passage: ")
                    except Exception:
                        continue
                    for j, emb in enumerate(batch_embs):
                        score = sum(a * b for a, b in zip(q_emb, emb))
                        chunk_scores.append((i + j, score))

                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                top_scores = [s[1] for _, s in chunk_scores[:5]]
                if top_scores:
                    query_scores.append(
                        {
                            "question": q["question"],
                            "top_score": round(top_scores[0], 4),
                            "avg_top5": round(sum(top_scores) / len(top_scores), 4),
                        }
                    )

            if query_scores:
                avg_top = sum(s["top_score"] for s in query_scores) / len(query_scores)
                avg_top5 = sum(s["avg_top5"] for s in query_scores) / len(query_scores)
            else:
                avg_top = 0
                avg_top5 = 0

            results.append(
                {
                    "model": emb_config["name"],
                    "dim": emb_config["dim"],
                    "avg_top1_semantic_score": round(avg_top, 4),
                    "avg_top5_semantic_score": round(avg_top5, 4),
                    "query_count": len(query_scores),
                }
            )

        return results


def print_ablation_table(results: list[dict[str, Any]]) -> None:
    """打印消融实验对比表格"""
    print("\n" + "=" * 70)
    print("  📊 消融实验对比")
    print("=" * 70)

    header = f"{'变体':<32} {'Hit Rate':>10} {'MRR':>8} {'NDCG@5':>8} {'语义分':>8} {'时间':>6}"
    print(header)
    print("-" * 70)

    for r in results:
        m = r["metrics"]["overall"]
        ss = r["metrics"].get("semantic_score", 0)
        avg_time = sum(rec.get("time_seconds", 0) for rec in r["records"]) / max(len(r["records"]), 1)
        print(
            f"{r['variant']:<32} {m['hit_rate']:>9.0%} {m['mrr']:>7.3f} "
            f"{m['ndcg_at_5']:>7.3f} {ss:>7.4f} {avg_time:>5.2f}s"
        )

    print("-" * 70)
    print("\n结论: 对比不同变体下的检索质量差异，定位瓶颈所在。")
    print("      若 full 与 vector_only 无差异 → rewrite 和 reranker 未起效")
    print("      若 Hit Rate 很低 → embedding 或 chunk 策略需优化\n")


def print_embedding_table(results: list[dict[str, Any]]) -> None:
    """打印 embedding 对比表格"""
    print("\n" + "=" * 70)
    print("  📊 Embedding 模型语义相似度对比")
    print("=" * 70)

    if not results:
        print("  ❌ 无可用结果")
        return

    header = f"{'模型':<28} {'维度':>6} {'Avg Top-1':>10} {'Avg Top-5':>10} {'评估题数':>8}"
    print(header)
    print("-" * 70)

    for r in results:
        print(
            f"{r['model']:<28} {r['dim']:>6} {r['avg_top1_semantic_score']:>9.4f} "
            f"{r['avg_top5_semantic_score']:>9.4f} {r['query_count']:>7}"
        )

    print("-" * 70)
    print("\n结论: Top-1 语义分表示 query 与最佳匹配 chunk 的余弦相似度。")
    print("      分数越高说明 embedding 越能捕捉语义对应关系。")
    print("      当前 multilingual-e5-small 得分为 ~0.03，正常范围应在 0.3-0.6。\n")


def main():
    parser = argparse.ArgumentParser(description="检索组件 A/B 对比评估")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--compare-embeddings", action="store_true", help="对比不同 Embedding 模型")
    parser.add_argument("--output", type=str, default="", help="结果保存路径（JSON）")

    args = parser.parse_args()

    # 默认运行全部
    run_ablation = args.ablation
    run_emb = args.compare_embeddings
    if not run_ablation and not run_emb:
        run_ablation = True
        run_emb = True

    evaluator = RetrievalEvaluator()
    questions = get_test_questions()
    report = {}

    if run_ablation:
        print("\n  🔬 阶段一：消融实验")
        print("  ====================")
        ablation_results = evaluator.run_ablation(questions)
        print_ablation_table(ablation_results)
        report["ablation"] = ablation_results

    if run_emb:
        print("\n  🔬 阶段二：Embedding 模型对比")
        print("  ====================")
        emb_results = evaluator.run_embedding_comparison(questions)
        print_embedding_table(emb_results)
        report["embedding_comparison"] = emb_results

    # 保存
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.abspath("eval_results"), f"retrieval_ablation_{timestamp}.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 结果已保存: {output_path}")


if __name__ == "__main__":
    main()

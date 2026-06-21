"""
评估 Pipeline 入口
支持参数对比评估和单轮评估，输出结构化报告

新增功能:
  --ablation:   消融实验（逐一开关 rewrite / reranker / hybrid）
  --compare-embeddings: 对比不同 Embedding 模型的语义相似度
  --plot:        自动生成可视化图表
  --report:      生成优化建议报告
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import compute_relevance
from src.rag_pipeline import RAGPipeline

from .evaluate_retrieval import RetrievalEvaluator, print_ablation_table, print_embedding_table
from .judge import judge_answer, judge_default_score
from .metrics import compute_all_metrics, print_metrics_report
from .test_questions import get_test_questions, print_question_summary


class Evaluator:
    """RAG 系统评估器"""

    def __init__(
        self,
        chunk_sizes: list[int] = None,
        top_k_values: list[int] = None,
        output_dir: str = "eval_results",
        use_judge: bool = False,
        data_dir: str = "data",
        enable_plot: bool = False,
        enable_report: bool = False,
    ):
        self.chunk_sizes = chunk_sizes or [300, 500, 800]
        self.top_k_values = top_k_values or [3, 5, 8]
        self.output_dir = os.path.abspath(output_dir)
        self.use_judge = use_judge
        self.data_dir = data_dir
        self.enable_plot = enable_plot
        self.enable_report = enable_report

        os.makedirs(self.output_dir, exist_ok=True)

        # 临时评估用向量数据库目录
        self.chroma_dir = os.path.abspath("chroma_db_eval")
        self._clean_chroma()

    def _clean_chroma(self):
        """清空评估用向量数据库"""
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

    def run(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        """运行评估

        Args:
            questions: 测试问题列表

        Returns:
            完整的评估报告 dict
        """
        print("=" * 70)
        print("  🚀 RAG 系统评估 Pipeline")
        print("=" * 70)
        print(f"  Chunk Sizes: {self.chunk_sizes}")
        print(f"  Top-K Values: {self.top_k_values}")
        print(f"  LLM Judge: {'开启' if self.use_judge else '关闭（规则评估）'}")
        print_question_summary(questions)

        detailed_results = []
        config_summaries = []

        pipeline = None
        for chunk_size in self.chunk_sizes:
            min_c = max(200, chunk_size - 200)
            max_c = chunk_size

            print("\n" + "=" * 70)
            print(f"  📊 评估 Chunk Size: {min_c}-{max_c} 字")
            print("=" * 70)

            # 释放旧 pipeline 的 ChromaDB 文件锁（不同 chunk size 的集合名不同，
            # force_reindex=True 会自动 delete_collection，无需 rmtree 整个目录）
            if pipeline is not None:
                pipeline.close()
                pipeline = None
            pipeline = RAGPipeline(
                data_dir=self.data_dir,
                persist_dir=self.chroma_dir,
                chunk_min_chars=min_c,
                chunk_max_chars=max_c,
            )

            count = pipeline.initialize_knowledge_base(force_reindex=True)
            if count == 0:
                print("  ❌ 知识库初始化失败，跳过")
                continue

            for top_k in self.top_k_values:
                print(f"\n  ▶ Top-K = {top_k}")
                pipeline.top_k = top_k

                config_key = f"chunk={min_c}-{max_c}_topk={top_k}"
                config_records = []

                for q in questions:
                    question = q["question"]
                    category = q.get("category", "unknown")

                    # ── 执行查询 ──
                    start = time.time()
                    result = pipeline.query(question, top_k=top_k)
                    elapsed = time.time() - start

                    # ── 提取指标 ──
                    sources = result.get("sources", [])
                    has_sources = len(sources) > 0

                    def _vec_score(s):
                        val = s.get("_vector_score")
                        return val if val is not None else s.get("score", 0)

                    top_score = _vec_score(sources[0]) if sources else 0
                    avg_score = sum(_vec_score(c) for c in sources) / len(sources) if sources else 0

                    # 检索耗时（从 sources 中无法得知，用总耗时的大致估算）
                    retrieval_time = max(elapsed * 0.15, 0.1)  # 估算：检索约占 15%

                    # 命中预期文档：兼容 SmartChunker 去掉后缀和 pipeline 保留后缀的情况
                    expected_hit = False
                    expected_doc = q.get("expected_doc", "")
                    if expected_doc:
                        expected_base = expected_doc.rsplit(".", 1)[0]  # 去掉后缀
                        expected_hit = any(
                            expected_doc == s["metadata"].get("filename", "")
                            or s["metadata"].get("filename", "") == expected_base
                            or s["metadata"]
                            .get("filename", "")
                            .startswith(expected_base.split("_", 1)[-1] if "_" in expected_base else expected_base)
                            for s in sources
                        )

                    # 相关性判断
                    relevance_info = compute_relevance(question, sources)
                    is_refusal = result.get("is_refusal", False)

                    # 用 pipeline 的拒答信号覆盖自己算的相关性（LLM 改写判断为 OOD 时优先）
                    pipeline_relevance = result.get("relevance", {})
                    if is_refusal and not relevance_info["is_relevant"]:
                        # 双方一致拒答 → 没问题
                        pass
                    elif is_refusal and relevance_info["is_relevant"]:
                        # pipeline 拒答但 overlap/语义判断为相关 → 信任 pipeline
                        relevance_info["is_relevant"] = False
                        relevance_info["reason"] = f"pipeline 拒答覆盖（原：{relevance_info['reason']}）"
                    elif (
                        not is_refusal
                        and not relevance_info["is_relevant"]
                        and pipeline_relevance.get("is_relevant", True)
                    ):
                        # pipeline 认为相关但规则认为不相关 → 信任规则（可能是 OOD 边界漏网）
                        pass

                    # 拒答正确性
                    if category == "out_of_knowledge":
                        refusals = [
                            not relevance_info["is_relevant"],
                            is_refusal,
                            relevance_info.get("overlap", 0) < 0.02,
                        ]
                        correct_refusal = sum(refusals) >= 2
                    else:
                        wrong_refusal = not relevance_info["is_relevant"] and is_refusal
                        correct_refusal = not wrong_refusal

                    record = {
                        "chunk_size": f"{min_c}-{max_c}",
                        "top_k": top_k,
                        "question": question,
                        "category": category,
                        "difficulty": q.get("difficulty", "unknown"),
                        "expected_doc": expected_doc,
                        "expected_hit": expected_hit,
                        "num_chunks": count,
                        "num_retrieved": len(sources),
                        "top_score": round(top_score, 4),
                        "avg_score": round(avg_score, 4),
                        "is_relevant": relevance_info["is_relevant"],
                        "overlap": round(relevance_info["overlap"], 4),
                        "relevance_reason": relevance_info["reason"],
                        "is_refusal_in_answer": is_refusal,
                        "correct_refusal": correct_refusal,
                        "time_seconds": round(elapsed, 2),
                        "retrieval_time": round(retrieval_time, 2),
                        "sources": sources,  # 新增：保存全量 sources
                        "answer_text": result.get("answer", ""),  # 新增：保存回答文本
                    }

                    # ── LLM Judge（可选） ──
                    if self.use_judge:
                        try:
                            judge_scores = judge_answer(
                                question=question,
                                answer=result.get("answer", ""),
                                sources=sources,
                                relevance_info=relevance_info,
                            )
                        except Exception:
                            judge_scores = judge_default_score()
                        record["judge_scores"] = judge_scores
                    else:
                        record["judge_scores"] = judge_default_score()

                    config_records.append(record)

                    # 打印进度
                    status = "✅" if correct_refusal else "❌"
                    if category == "exact_match":
                        status = "✅" if expected_hit else "⚠️"
                    print(
                        f"    {status} [{category[:4]}] [{q.get('difficulty', '?')[:4]}] {question[:30]:<30s} "
                        f"score={top_score:.3f} time={elapsed:.1f}s"
                    )

                # ── 计算该配置的指标 ──
                config_metrics = compute_all_metrics(config_records)
                config_summaries.append(
                    {
                        "config": config_key,
                        "metrics": config_metrics,
                        "record_count": len(config_records),
                    }
                )
                detailed_results.extend(config_records)

        # ── 总报告 ──
        report = self._build_report(detailed_results, config_summaries)

        # 关闭最后一个 pipeline，释放 ChromaDB 文件锁
        if pipeline is not None:
            pipeline.close()

        return report

    def _build_report(
        self,
        detailed_results: list[dict[str, Any]],
        config_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构建最终评估报告"""
        # 总体指标
        overall_metrics = compute_all_metrics(detailed_results)

        # 最优配置（综合评分）
        best_config = None
        best_score = 0
        for cs in config_summaries:
            m = cs["metrics"]["overall"]
            composite = (
                m.get("hit_rate", 0) * 0.30
                + m.get("refusal_accuracy", 0) * 0.25
                + m.get("average_precision", 0) * 0.15
                + cs["metrics"].get("semantic_score", 0) * 0.15
                + min(cs["metrics"].get("passage_diversity", 0) / 5.0, 1.0) * 0.15
            )
            if composite > best_score:
                best_score = composite
                best_config = cs["config"]

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "chunk_sizes": self.chunk_sizes,
                "top_k_values": self.top_k_values,
                "use_judge": self.use_judge,
                "total_queries": len(detailed_results),
                "config_count": len(config_summaries),
            },
            "config_summaries": config_summaries,
            "best_config": best_config,
            "best_composite_score": round(best_score, 4),
            "metrics": overall_metrics,
            "detailed_results": detailed_results,
        }

        return report

    def save_report(self, report: dict[str, Any], filename: str | None = None) -> str:
        """保存评估报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  📄 报告已保存: {filepath}")
        return filepath


def print_summary_report(report: dict[str, Any]) -> None:
    """打印评估摘要报告"""
    print("\n" + "=" * 70)
    print("  📊 评估报告摘要")
    print("=" * 70)

    metadata = report.get("report_metadata", {})
    print(f"\n  配置: Chunk {metadata.get('chunk_sizes')} × Top-K {metadata.get('top_k_values')}")
    print(f"  总查询: {metadata.get('total_queries')} 次")

    # 打印指标
    print_metrics_report(report.get("metrics", {}))

    # 最优配置
    if report.get("best_config"):
        print(f"🏆 推荐配置: {report['best_config']} (综合评分: {report.get('best_composite_score', 0):.4f})")
        print("   评分公式: Hit Rate×0.30 + Refusal Acc×0.25 + Avg Prec×0.15 + Semantic×0.15 + Diversity×0.15")
        print()

    # 各配置对比表格
    summaries = report.get("config_summaries", [])
    if summaries:
        print(f"{'配置':<28} {'Hit Rate':>10} {'MRR':>8} {'NDCG@5':>8} {'Refusal':>8} {'语义分':>8} {'Count':>6}")
        print("-" * 78)
        for cs in summaries:
            m = cs["metrics"]["overall"]
            ss = cs["metrics"].get("semantic_score", 0)
            print(
                f"{cs['config']:<28} {m.get('hit_rate', '?'):>9.0%} "
                f"{m.get('mrr', 0):>7.3f} {m.get('ndcg_at_5', 0):>7.3f} "
                f"{m.get('refusal_accuracy', '?'):>7.0%} {ss:>7.4f} {cs['record_count']:>5}"
            )
        print("-" * 78)

    # Judge 评分平均
    judge_scores_all = []
    for r in report.get("detailed_results", []):
        js = r.get("judge_scores", {})
        if js and js.get("mode") not in (None, "none"):
            judge_scores_all.append(js)
    if judge_scores_all:
        avg = {
            dim: round(sum(js.get(dim, 0) for js in judge_scores_all) / len(judge_scores_all), 2)
            for dim in ["faithfulness", "completeness", "helpfulness", "citation_accuracy", "overall"]
        }
        print(f"\n🤖 LLM-as-Judge 平均分（{len(judge_scores_all)} 次评估）:")
        print(
            f"   忠实度: {avg['faithfulness']:.1f} | 完整性: {avg['completeness']:.1f} | "
            f"有用性: {avg['helpfulness']:.1f} | 引用准确: {avg['citation_accuracy']:.1f} | "
            f"综合: {avg['overall']:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评估 Pipeline")
    parser.add_argument(
        "--chunk-sizes", type=int, nargs="+", default=[300, 500, 800], help="要测试的 Chunk Size（默认: 300 500 800）"
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 8], help="要测试的 Top-k 值（默认: 3 5 8）")
    parser.add_argument("--output", type=str, default="", help="评估结果输出文件名（默认自动生成）")
    parser.add_argument("--quick", action="store_true", help="快速模式：仅 Chunk 500 + Top-k 3/5/8")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge 回答质量评估")
    parser.add_argument("--data-dir", type=str, default="data", help="文档目录（默认: data）")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验（与 grid search 配合）")
    parser.add_argument("--compare-embeddings", action="store_true", help="对比不同 Embedding 模型的语义相似度")
    parser.add_argument("--plot", action="store_true", help="自动生成可视化图表")
    parser.add_argument("--report", action="store_true", help="生成优化建议报告")

    args = parser.parse_args()

    chunk_sizes = [500] if args.quick else args.chunk_sizes

    # 创建评估器
    evaluator = Evaluator(
        chunk_sizes=chunk_sizes,
        top_k_values=args.top_k,
        use_judge=args.judge,
        data_dir=args.data_dir,
        enable_plot=args.plot,
        enable_report=args.report,
    )

    # 1. 加载测试问题
    questions = get_test_questions()

    # 2. 运行核心评估
    report = evaluator.run(questions)

    # 3. 运行消融实验（可选）
    if args.ablation:
        print("\n  🔬 运行消融实验...")
        ret_eval = RetrievalEvaluator(data_dir=args.data_dir)
        ablation_results = ret_eval.run_ablation(questions)
        report["ablation"] = ablation_results
        print_ablation_table(ablation_results)

    # 4. Embedding 对比（可选）
    if args.compare_embeddings:
        print("\n  🔬 运行 Embedding 模型对比...")
        ret_eval = RetrievalEvaluator(data_dir=args.data_dir)
        emb_results = ret_eval.run_embedding_comparison(questions)
        report["embedding_comparison"] = emb_results
        print_embedding_table(emb_results)

    # 5. 保存报告
    saved_path = evaluator.save_report(report, args.output if args.output else None)

    # 6. 打印摘要
    print_summary_report(report)

    # 7. 可视化（可选）
    if args.plot:
        try:
            from .visualize import generate_all_plots

            generate_all_plots(report)
        except ImportError as e:
            print(f"  ⚠️ 可视化导入失败: {e}（需要 matplotlib）")
        except Exception as e:
            print(f"  ⚠️ 可视化生成失败: {e}")

    # 8. 优化建议报告（可选）
    if args.report:
        try:
            from .report import generate_report

            report_text = generate_report(
                metrics=report.get("metrics", {}),
                ablation_results=report.get("ablation"),
                embedding_results=report.get("embedding_comparison"),
            )
            # 保存到同目录
            report_path = saved_path.replace(".json", "_recommendations.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"\n  📄 优化建议报告: {report_path}")
        except Exception as e:
            print(f"  ⚠️ 优化建议报告生成失败: {e}")

    # 兼容旧版：同时保存 detailed_results 到根目录
    compat_path = os.path.join(os.path.abspath("."), "evaluation_report.json")
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(report.get("detailed_results", []), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 兼容报告: {compat_path}")


if __name__ == "__main__":
    main()

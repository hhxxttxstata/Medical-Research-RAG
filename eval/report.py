"""
优化建议报告生成

读取评估结果的 JSON 报告，自动诊断当前 RAG 系统的瓶颈，生成
中文化的优化建议报告，包括：
  - 当前各组件表现诊断
  - 推荐优化方向
  - 预期提升幅度
  - 优先级排序

直接可用于秋招项目展示中的"持续优化"章节。
"""

import json
import os
from typing import Any

# ── 诊断阈值 ──

THRESHOLDS = {
    "semantic_score": {
        "good": 0.30,
        "fair": 0.15,
        "poor": 0.05,
    },
    "hit_rate": {
        "good": 0.90,
        "fair": 0.70,
        "poor": 0.50,
    },
    "refusal_accuracy": {
        "good": 0.95,
        "fair": 0.85,
        "poor": 0.70,
    },
    "passage_diversity": {
        "good": 3.0,
        "fair": 2.0,
        "poor": 1.0,
    },
    "answer_efficiency": {
        "good": 0.60,
        "fair": 0.30,
        "poor": 0.10,
    },
}


def _grade(value: float, metric: str) -> str:
    """根据阈值评分"""
    t = THRESHOLDS.get(metric, {"good": 0.8, "fair": 0.5, "poor": 0.2})
    if value >= t["good"]:
        return "✅ 优秀"
    elif value >= t["fair"]:
        return "⚠️ 待优化"
    elif value >= t["poor"]:
        return "🔴 需改进"
    return "❌ 严重问题"


# ── 诊断报告生成 ──


def generate_report(
    metrics: dict[str, Any],
    ablation_results: list[dict[str, Any]] | None = None,
    embedding_results: list[dict[str, Any]] | None = None,
) -> str:
    """生成优化建议报告

    Args:
        metrics: compute_all_metrics() 的输出
        ablation_results: evaluate_retrieval 消融实验输出
        embedding_results: embedding 对比结果

    Returns:
        中文化报告文本
    """
    overall = metrics.get("overall", {})
    by_diff = metrics.get("by_difficulty", {})
    refusal_detail = metrics.get("refusal_detail", {})
    semantic = metrics.get("semantic_score", 0)
    diversity = metrics.get("passage_diversity", 0)
    efficiency = metrics.get("answer_efficiency", 0)
    fpr = metrics.get("false_positive_rate", 0)
    tb = metrics.get("time_breakdown", {})

    lines = []
    lines.append("=" * 60)
    lines.append("  📋 RAG 系统优化建议报告")
    lines.append("=" * 60)

    # ── 概览 ──
    lines.append("\n## 一、系统概览")
    lines.append("\n  检索质量:")
    hr = overall.get("hit_rate", 0)
    lines.append(f"    Hit Rate:       {hr:.1%}  [{_grade(hr, 'hit_rate')}]")
    mrr_val = overall.get("mrr", 0)
    lines.append(f"    MRR:            {mrr_val:.4f}")
    lines.append(f"    Semantic Score: {semantic:.4f}  [{_grade(semantic, 'semantic_score')}]")
    lines.append(f"    Passage Div.:   {diversity:.2f} docs/q  [{_grade(diversity, 'passage_diversity')}]")
    lines.append(f"    Answer Effic.:  {efficiency:.4f}  [{_grade(efficiency, 'answer_efficiency')}]")

    ra = overall.get("refusal_accuracy", 0)
    lines.append("\n  拒答质量:")
    lines.append(f"    Refusal Acc.:   {ra:.1%}  [{_grade(ra, 'refusal_accuracy')}]")
    lines.append(f"    False Pos. Rate:{fpr:.1%}")
    lines.append(f"    Precision:      {refusal_detail.get('precision', 0):.1%}")
    lines.append(f"    Recall:         {refusal_detail.get('recall', 0):.1%}")
    lines.append(f"    F1:             {refusal_detail.get('f1', 0):.1%}")

    lines.append("\n  响应时间:")
    lines.append(f"    平均总耗时:    {tb.get('avg_total_time', 0):.2f}s")
    lines.append(f"    平均检索耗时:  {tb.get('avg_retrieval_time', 0):.2f}s")
    lines.append(f"    生成占比:      {tb.get('generation_ratio', 0):.1%}")

    # ── 瓶颈诊断 ──
    lines.append("\n## 二、瓶颈诊断")

    recommendations = []

    # 诊断 1: Embedding 质量
    if semantic < THRESHOLDS["semantic_score"]["fair"]:
        rec = {
            "component": "Embedding 模型",
            "issue": f"语义相似度仅 {semantic:.4f}，远低于正常范围 (0.3-0.6)",
            "suggestion": "建议更换为更高精度模型：\n"
            f"   - multilingual-e5-base (768维，预期提升至 0.15-0.30)\n"
            f"   - BAAI/bge-m3 (1024维，预期提升至 0.30-0.50)\n"
            f"   - text-embedding-3-small (1536维，预期提升至 0.40-0.70)\n"
            f"  预期: top_score 可从 {semantic:.3f} 提升至 0.3-0.5",
            "priority": "P0 — 最关键",
        }
        recommendations.append(rec)
        lines.append(f"\n  🔴 P0 — {rec['component']}")
        lines.append(f"    问题: {rec['issue']}")
        lines.append(f"    建议: {rec['suggestion']}")

    # 诊断 2: Chunk 策略
    if diversity < THRESHOLDS["passage_diversity"]["fair"]:
        rec = {
            "component": "Chunk 大小/策略",
            "issue": f"每个查询平均仅检索到 {diversity:.1f} 个不同文档，多样性不足",
            "suggestion": "可能原因：chunk 太小或重叠不足，建议：\n"
            "   - 增大 chunk size (500→800)\n"
            "   - 增加 chunk 间重叠 (50→100)\n"
            "   - 启用 Small-to-Big 策略提升上下文完整性",
            "priority": "P1 — 重要",
        }
        recommendations.append(rec)
        lines.append(f"\n  🟡 P1 — {rec['component']}")
        lines.append(f"    问题: {rec['issue']}")
        lines.append(f"    建议: {rec['suggestion']}")

    # 诊断 3: 消融实验分析
    if ablation_results:
        # 找出 full 和 vector_only 的差异
        full = next((a for a in ablation_results if a["variant"].startswith("full")), None)
        vec = next((a for a in ablation_results if "vector_only" in a["variant"]), None)

        if full and vec:
            full_hr = full["metrics"]["overall"]["hit_rate"]
            vec_hr = vec["metrics"]["overall"]["hit_rate"]
            if abs(full_hr - vec_hr) < 0.05:
                rec = {
                    "component": "Query Rewriting / Reranker",
                    "issue": "消融实验显示 rewrite + reranker 与纯向量检索差异较小",
                    "suggestion": "Rewrite 和 Reranker 可能未起效，建议检查：\n"
                    "   - Rewrite 提示词是否适配医学领域\n"
                    "   - Reranker 评分是否过于均匀（缺乏区分度）\n"
                    "   - 或关闭两者以降低延迟（省 ~2-5s/查询）",
                    "priority": "P2 — 可优化",
                }
                recommendations.append(rec)
                lines.append(f"\n  🟢 P2 — {rec['component']}")
                lines.append(f"    问题: {rec['issue']}")
                lines.append(f"    建议: {rec['suggestion']}")

    # 诊断 4: 拒答能力
    if fpr > 0.1:
        lines.append(f"\n  🟡 拒答误判率 {fpr:.1%}，尝试调整 relevance 阈值")
    elif ra > 0.95:
        lines.append(f"\n  🟢 拒答能力优秀（准确率 {ra:.1%}），当前阈值设置合理")

    # 诊断 5: 按难度分层
    if by_diff:
        lines.append("\n  🎯 按难度分析:")
        for diff in ["hard", "medium", "easy"]:
            dm = by_diff.get(diff)
            if dm:
                lines.append(
                    f"     {diff:<8}: {dm['count']}题, "
                    f"Hit Rate={dm['hit_rate']:.0%}, "
                    f"语义分={dm['avg_semantic_score']:.4f}"
                )

    # ── 推荐优先级总表 ──
    lines.append("\n## 三、推荐优化优先级")
    lines.append(f"\n  {'优先级':<12} {'组件':<20} {'预期提升'}")
    lines.append("  " + "-" * 50)

    for rec in sorted(recommendations, key=lambda r: r["priority"]):
        lines.append(f"  {rec['priority']:<12} {rec['component']:<20} {rec.get('suggestion', '')[:60]}...")

    # ── 面试亮点 ──
    lines.append("\n## 四、面试中可以讲的技术亮点")
    lines.append("\n  1. 构建了完整的 RAG 评估体系（检索指标 + 生成质量 + 拒答能力）")
    lines.append("  2. 通过消融实验验证各组件贡献，数据驱动的优化决策")
    lines.append(f"  3. 发现当前 {semantic:.4f} 语义相似度瓶颈，定位到 embedding 模型")
    lines.append("  4. 按难度分层评估（easy/medium/hard），精细化衡量系统边界")
    lines.append("  5. 可视化对比多配置，面试时一目了然展示优化历程")

    lines.append("\n" + "=" * 60)
    lines.append("  报告结束")
    lines.append("=" * 60 + "\n")

    return "\n".join(lines)


def generate_report_from_json(json_path: str) -> str:
    """从评估 JSON 文件生成报告"""
    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    metrics = report.get("metrics", {})
    if not metrics:
        # 兼容旧报告格式：只含 detailed_results
        from .metrics import compute_all_metrics

        detailed = report.get("detailed_results", report if isinstance(report, list) else [])
        metrics = compute_all_metrics(detailed)

    return generate_report(
        metrics=metrics,
        ablation_results=report.get("ablation"),
        embedding_results=report.get("embedding_comparison"),
    )


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 优化建议报告生成")
    parser.add_argument("json_path", type=str, nargs="?", default="", help="评估结果 JSON 文件路径")
    parser.add_argument("--output", type=str, default="", help="报告保存路径（默认打印到控制台）")

    args = parser.parse_args()

    if args.json_path:
        report_text = generate_report_from_json(args.json_path)
    else:
        # 无输入时尝试读取最新评估报告
        eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results")
        if os.path.isdir(eval_dir):
            reports = sorted(
                [f for f in os.listdir(eval_dir) if f.startswith("eval_report_") and f.endswith(".json")],
                reverse=True,
            )
            if reports:
                latest = os.path.join(eval_dir, reports[0])
                print(f"  自动读取最新报告: {latest}")
                report_text = generate_report_from_json(latest)
            else:
                print("  ⚠️ 未找到评估报告 JSON 文件")
                return
        else:
            print("  ⚠️ 未找到 eval_results 目录")
            return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"  📄 报告已保存: {args.output}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()

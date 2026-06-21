"""
评估结果可视化

生成图表用于：
  - 平行坐标图：多配置对比（适合 grid search / 消融实验）
  - 雷达图：每个配置的多维能力对比
  - 柱状图：按难度分层的指标
  - 热力图：配置参数与指标的相关性

图表保存到 eval_results/figures/ 目录
可直接用于秋招项目展示
"""

import os
from typing import Any

# matplotlib 非交互式后端，避免弹出窗口
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results", "figures")


def _ensure_output_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ── 颜色方案 ──

COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]


# ── 雷达图 ──


def plot_radar(
    configs: list[dict[str, Any]],
    save_name: str = "radar_comparison.png",
    title: str = "RAG 系统配置多维对比",
):
    """生成各配置的雷达图

    Args:
        configs: [{"name": "配置A", "metrics": {"hit_rate": ..., "mrr": ..., ...}}, ...]
        save_name: 保存文件名
        title: 图表标题
    """
    _ensure_output_dir()
    if not configs:
        return

    # 选取指标维度
    dims = ["hit_rate", "mrr", "ndcg_at_5", "refusal_accuracy", "average_precision"]
    dim_labels = ["Hit Rate", "MRR", "NDCG@5", "Refusal Acc.", "Avg Prec."]
    num_dims = len(dims)

    # 准备数据（归一化到 0-1）
    data = {}
    for cfg in configs:
        m = cfg.get("metrics", {})
        if isinstance(m, dict) and "overall" in m:
            m = m["overall"]
        data[cfg.get("name", cfg.get("config", "?"))] = [m.get(d, 0) for d in dims]

    if not data:
        return

    # 画布
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (name, values) in enumerate(data.items()):
        values_closed = values + values[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, values_closed, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values_closed, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    filepath = os.path.join(FIGURES_DIR, save_name)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 雷达图: {filepath}")


# ── 柱状图：按难度分层 ──


def plot_difficulty_bar(
    by_difficulty: dict[str, dict[str, float]],
    save_name: str = "difficulty_analysis.png",
    title: str = "按难度分层的检索质量",
):
    """按难度层级绘制指标柱状图"""
    _ensure_output_dir()
    if not by_difficulty:
        return

    diffs = ["easy", "medium", "hard"]
    metrics_to_plot = ["hit_rate", "refusal_accuracy"]
    metric_labels = ["Hit Rate", "Refusal Acc."]

    x = np.arange(len(diffs))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [by_difficulty.get(d, {}).get(metric, 0) for d in diffs if d in by_difficulty]
        # 只取存在的难度
        existing_diffs = [d for d in diffs if d in by_difficulty]
        x_pos = np.arange(len(existing_diffs))
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        bars = ax.bar(
            x_pos + offset,
            [by_difficulty[d].get(metric, 0) for d in existing_diffs],
            width,
            label=label,
            color=COLORS[i],
        )

    ax.set_xticks(np.arange(len([d for d in diffs if d in by_difficulty])))
    ax.set_xticklabels([d.capitalize() for d in diffs if d in by_difficulty])
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    filepath = os.path.join(FIGURES_DIR, save_name)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 难度分析图: {filepath}")


# ── 平行坐标图 ──


def plot_parallel_coordinates(
    configs: list[dict[str, Any]],
    save_name: str = "parallel_coords.png",
    title: str = "多配置平行坐标对比",
):
    """生成平行坐标图比较多个配置

    Args:
        configs: [{"name": ..., "metrics": {"hit_rate": ..., "mrr": ..., ...}}, ...]
    """
    _ensure_output_dir()
    if len(configs) < 2:
        return

    dims = ["hit_rate", "mrr", "ndcg_at_5", "refusal_accuracy", "average_precision"]
    dim_labels = ["Hit Rate", "MRR", "NDCG@5", "Refusal Acc.", "Avg Prec."]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(dims))

    for i, cfg in enumerate(configs):
        m = cfg.get("metrics", {})
        if isinstance(m, dict) and "overall" in m:
            m = m["overall"]
        values = [m.get(d, 0) for d in dims]
        color = COLORS[i % len(COLORS)]
        ax.plot(
            x,
            values,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=cfg.get("name", cfg.get("config", f"Config {i + 1}")),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    filepath = os.path.join(FIGURES_DIR, save_name)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 平行坐标图: {filepath}")


# ── 热力图：配置参数指标相关性 ──


def plot_config_heatmap(
    configs: list[dict[str, Any]],
    save_name: str = "config_heatmap.png",
    title: str = "配置 × 指标 热力图",
):
    """绘制配置参数与指标的热力图"""
    _ensure_output_dir()
    if not configs:
        return

    dims = ["hit_rate", "mrr", "ndcg_at_5", "refusal_accuracy", "average_precision"]
    dim_labels = ["Hit Rate", "MRR", "NDCG@5", "Refusal Acc.", "Avg Prec."]

    # 提取数据矩阵
    names = []
    matrix = []
    for cfg in configs:
        m = cfg.get("metrics", {})
        if isinstance(m, dict) and "overall" in m:
            m = m["overall"]
        names.append(cfg.get("name", cfg.get("config", "?")))
        matrix.append([m.get(d, 0) for d in dims])

    if not matrix:
        return

    fig, ax = plt.subplots(figsize=(len(dims) * 1.2 + 2, len(names) * 0.6 + 2))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)

    # 标注数值
    for i in range(len(names)):
        for j in range(len(dim_labels)):
            val = matrix[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black" if val > 0.5 else "white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

    filepath = os.path.join(FIGURES_DIR, save_name)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 热力图: {filepath}")


# ── 全量生成 ──


def generate_all_plots(report: dict[str, Any], prefix: str = ""):
    """从评估报告 JSON 生成所有图表"""
    _ensure_output_dir()

    # 1. 如果报告包含 config_summaries → 雷达图 + 平行坐标图
    configs = report.get("config_summaries", [])
    if configs:
        # 为每个配置添加 metrics 结构
        plot_radar(configs, save_name=f"{prefix}radar_comparison.png")
        if len(configs) >= 2:
            plot_parallel_coordinates(configs, save_name=f"{prefix}parallel_coords.png")
            plot_config_heatmap(configs, save_name=f"{prefix}config_heatmap.png")

    # 2. 如果报告包含 by_difficulty → 难度分析图
    metrics = report.get("metrics", {})
    by_diff = metrics.get("by_difficulty", {})
    if by_diff:
        plot_difficulty_bar(by_diff, save_name=f"{prefix}difficulty_analysis.png")

    # 3. 如果有 ablation 结果 → 专门对比图
    ablation = report.get("ablation", [])
    if ablation:
        plot_parallel_coordinates(
            [{"name": a["variant"], "metrics": a["metrics"]} for a in ablation],
            save_name=f"{prefix}ablation_comparison.png",
            title="消融实验对比",
        )

    print(f"\n  🎨 所有图表已保存到: {FIGURES_DIR}")

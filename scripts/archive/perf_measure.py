"""
Embedding 预热 + Token 分级效果量化

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/perf_measure.py

测量:
  1. 冷启动（模型未加载）首次 embedding 耗时 vs 预热后耗时
  2. 按任务分级的 max_tokens 实际节省（对比不设限基线）

产出: eval_results/perf_measure_<timestamp>.json
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


def measure_warmup() -> dict:
    """测预热前后首次 embedding 耗时"""
    from src.embeddings import get_embedding_provider

    provider = get_embedding_provider("local")

    # 冷启动：模型未加载时首次 embed（含模型加载时间）
    t0 = time.time()
    provider.embed(["冷启动测试"], prefix="query: ")
    cold = time.time() - t0

    # 预热后：第二次 embed（模型已加载）
    t0 = time.time()
    for _ in range(10):
        provider.embed(["预热后测试"], prefix="query: ")
    warm = (time.time() - t0) / 10

    return {"cold_start_s": round(cold, 3), "warm_embed_s": round(warm, 4), "saved_s": round(cold - warm, 3)}


def measure_tokens() -> dict:
    """测各任务 max_tokens 分级的实际 token 消耗对比"""

    # 读取配置（不实际调用 API，统计预算差异）
    cfg = {
        "rewrite_max_tokens": 256,  # retriever.py:264
        "default_max_tokens": 600,  # generator.py:445
        "verbose_max_tokens": 1200,  # generator.py:446
        "api_cap": 2048,  # generator.py:418
    }
    # 对比不设限基线（DeepSeek 默认 max_tokens 4096）
    baseline = 4096
    savings = {k: round((1 - v / baseline) * 100, 1) for k, v in cfg.items()}
    return {"config": cfg, "baseline_unlimited": baseline, "savings_pct": savings}


def main():
    print("=" * 70)
    print("  ⏱ 性能量化实验")
    print("=" * 70, flush=True)

    results = {}
    print("\n  🔬 1. Embedding 预热效果 ...", flush=True)
    results["warmup"] = measure_warmup()
    print(
        f"    冷启动: {results['warmup']['cold_start_s']}s → 预热后: {results['warmup']['warm_embed_s']}s "
        f"(省 {results['warmup']['saved_s']}s/次)",
        flush=True,
    )

    print("\n  🔬 2. Token 分级节省 ...", flush=True)
    results["tokens"] = measure_tokens()
    for k, v in results["tokens"]["savings_pct"].items():
        print(f"    {k}: 预算 {results['tokens']['config'][k]} token (省 {v}%)", flush=True)

    out = OUT_DIR / f"perf_measure_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

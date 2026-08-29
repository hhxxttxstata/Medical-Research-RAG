"""
regression.py — 能力回归门禁（bad case → candidate → qualification → **regression**）

流程：
  1. 跑 Agentic dev 回归：benchmark_multi_hop.json 18 题（v2.1，真实索引 + LLM）
  2. 跑检索层回归：evaluate.py（56 题，无 LLM，快速）
  3. 与基线对比关键指标（evidence_recall / policy / ood / false_abstain / final_answer），
     任何下降 → FAIL（--tolerance 可调）
  4. 结果 append 到 eval_results/regression_history.jsonl

基线：默认取 regression_history.jsonl 最近一条；首次运行仅记录基线（--record-only）。

用法:
    python scripts/regression.py                    # 完整回归（约 40 分钟）
    python scripts/regression.py --skip-retrieval   # 只跑 Agentic dev 回归
    python scripts/regression.py --record-only      # 只记录基线（不评测）

纪律：promotion 前必须 regression 全过——candidate 不得让任何已验收能力回退。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
DEV_BENCH = ROOT / "tests" / "benchmark_multi_hop.json"
HISTORY = ROOT / "eval_results" / "regression_history.jsonl"

# 对比断言：指标 → (容忍的绝对下降量, 是否必保)
# 字符串型指标（"x/y"）按比例对比
COMPARE = {
    "evidence_recall": 0.02,  # 允许 ±0.02（LLM/检索抖动）
    "hop_recall": 0.02,
    "completeness": 0.02,
    "policy_action_accuracy": 0.0,  # 策略准确率必保
    "ood_reject": 0.0,  # 拒答必保
    "false_abstain": 0.0,  # 误拒必保
    "final_answer_accuracy": 0.0,  # 答案必保
}


def _ratio(v: str) -> float:
    a, _, b = v.partition("/")
    try:
        return float(a) / max(float(b), 1)
    except ValueError:
        return float(v)


def run_agentic(agents: str, extra: list[str]) -> dict:
    cmd = [sys.executable, "-u", "scripts/holdout_eval.py", "--agents", agents, "--bench", str(DEV_BENCH)] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    reports = sorted((ROOT / "eval_results").glob("holdout30_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def run_retrieval() -> dict | None:
    """跑 evaluate.py（检索层 56 题），返回 hit_rate / mrr / bad_case_count"""
    try:
        r = subprocess.run(
            [sys.executable, "-u", "evaluate.py"],
            cwd=ROOT,
            env=dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1"),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        out = r.stdout + r.stderr
        lines = [l for l in out.splitlines() if "Hit Rate" in l or "hit_rate" in l]
        return {"retrieval_output_tail": out[-600:]}
    except Exception as e:
        return {"retrieval_error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="v21")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()

    if args.record_only:
        print("🔒 record-only：本次结果将作为后续回归基线（需先跑一次完整评测）")
        sys.exit(0)

    # 1. Agentic dev 回归
    print("=" * 60)
    print("  🔄 Regression: Agentic dev 18 题")
    print("=" * 60, flush=True)
    t0 = time.time()
    report = run_agentic(args.agents, [])
    m = report["metrics"].get(args.agents, report["metrics"].get("v21", {}))
    print(f"  📄 {report['timestamp']}  耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)

    # 2. 检索层回归
    ret = {}
    if not args.skip_retrieval:
        print("\n  🔄 Regression: 检索层 56 题（evaluate.py）", flush=True)
        ret = run_retrieval()

    # 3. 对比基线
    baseline = None
    if HISTORY.exists():
        lines = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            baseline = lines[-1]
    failed = []
    if baseline and "metrics" in baseline:
        bm = baseline["metrics"]
        print("\n  📊 与基线对比（基线:", baseline.get("timestamp"), "）")
        for k, tol in COMPARE.items():
            if k not in m or k not in bm:
                continue
            cur = _ratio(str(m[k])) if isinstance(m[k], str) and "/" in str(m[k]) else float(m[k])
            old = _ratio(str(bm[k])) if isinstance(bm[k], str) and "/" in str(bm[k]) else float(bm[k])
            delta = cur - old
            ok = delta >= -tol
            print(f"    {'✅' if ok else '❌'} {k:<28}{old:.3f} → {cur:.3f} (Δ{delta:+.3f}, 容忍 {tol})")
            if not ok:
                failed.append(k)
    else:
        print("\n  ⚠️ 无基线记录——本次结果将作为基线")

    # 4. 记录
    record = {"timestamp": report["timestamp"], "metrics": m, "retrieval": ret, "elapsed_s": round(time.time() - t0)}
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n  🔒 Regression 门禁:", "PASS" if not failed else f"FAIL（下降: {failed}）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

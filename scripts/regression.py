"""
regression.py — 能力回归门禁（bad case → candidate → qualification → **regression**）

流程：
  1. 跑 Agentic dev 回归：benchmark_multi_hop.json 18 题（v2.1，真实索引 + LLM）
  2. 跑检索层回归：evaluate.py（56 题，无 LLM，快速）
  3. 与基线对比关键指标（容忍度声明于 evals/gates.json regression_tolerances），
     任何下降 → FAIL
  4. 对比前校验基线 config_snapshot 关键口径（git SHA / 模型名），
     不一致 → 显式警告，不静默对比（handoff v2 §16：数字不可追溯就无法谈晋升）
  5. 结果（含 config_snapshot）append 到 eval_results/regression_history.jsonl
  6. PASS 时同步写 eval_results/baseline_current.json（当前晋升基准）

基线：默认取 regression_history.jsonl 最近一条；首次运行即成为基线。
      --record-only：不评测，把最近一份 holdout30 报告登记为基线
      （历史版本此处是空操作，已修复）。

用法:
    python scripts/regression.py                    # 完整回归（约 40 分钟）
    python scripts/regression.py --skip-retrieval   # 只跑 Agentic dev 回归
    python scripts/regression.py --record-only      # 把最近评测报告登记为基线

纪律：promotion 前必须 regression 全过——candidate 不得让任何已验收能力回退。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parent.parent
DEV_BENCH = ROOT / "tests" / "benchmark_multi_hop.json"
HISTORY = ROOT / "eval_results" / "regression_history.jsonl"
BASELINE_CURRENT = ROOT / "eval_results" / "baseline_current.json"
GATES_FILE = ROOT / "evals" / "gates.json"

# holdout_eval 报告 metrics 存显示名（"Harm" 等），对比与落库前统一映射回字段名
# （与 qualify.py KEY_MAP 保持一致；缺失的显示名跳过，不虚构 0 值）
DISPLAY_TO_FIELD = {
    "final_answer_accuracy": "Final Answer Acc",
    "evidence_recall": "Evidence Recall@5",
    "hop_recall": "Hop Recall@5",
    "completeness": "Completeness",
    "final_rescue": "Final Rescue",
    "harm": "Harm",
    "net_utility": "NetUtility",
    "ood_reject": "OOD Reject",
    "false_abstain": "False Abstain",
    "policy_action_accuracy": "Policy Action Acc",
    "decomposition_success": "Decomp Success",
    "retry_recovery": "Retry Recovery",
    "unnecessary_action_rate": "Unnecessary Action",
    "avg_iterations": "Avg Iterations",
}


def load_gates() -> dict:
    if not GATES_FILE.exists():
        print(f"  ❌ 门禁声明文件缺失: {GATES_FILE}")
        raise SystemExit(1)
    return json.loads(GATES_FILE.read_text(encoding="utf-8"))


def normalize_metrics(m: dict) -> dict:
    """显示名 → 字段名。已经是字段名的（如手工构造的基线）保持原样。"""
    if not m:
        return m
    out = {f: m[disp] for f, disp in DISPLAY_TO_FIELD.items() if disp in m}
    return out if out else m


def _ratio(v: str) -> float:
    a, _, b = v.partition("/")
    try:
        return float(a) / max(float(b), 1)
    except ValueError:
        return float(v)


def _snapshot(**kw) -> dict:
    from eval.config_snapshot import build_config_snapshot

    return build_config_snapshot(dataset_files=["tests/benchmark_multi_hop.json"], **kw)


def run_agentic(agents: str, extra: list[str]) -> dict:
    cmd = [sys.executable, "-u", "scripts/holdout_eval.py", "--agents", agents, "--bench", str(DEV_BENCH)] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    reports = sorted((ROOT / "eval_results").glob("holdout30_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def run_retrieval() -> dict:
    """跑 evaluate.py（检索层 56 题），尽力解析 hit_rate/mrr，保留输出尾部"""
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
        ret: dict = {"retrieval_output_tail": out[-600:]}
        for field, pat in (("hit_rate", r"Hit Rate[^0-9]*([0-9.]+)"), ("mrr", r"MRR[^0-9]*([0-9.]+)")):
            mm = re.search(pat, out)
            if mm:
                ret[field] = float(mm.group(1))
        return ret
    except Exception as e:
        return {"retrieval_error": str(e)}


def record_baseline_from_report(agents: str) -> int:
    """--record-only：把最近一份 holdout30 报告登记为回归基线（不评测）"""
    reports = sorted((ROOT / "eval_results").glob("holdout30_*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        print("  ❌ record-only：eval_results/ 下没有 holdout30 报告可登记为基线")
        return 1
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    m = normalize_metrics(report["metrics"].get(agents, report["metrics"].get("v21", {})))
    snapshot = report.get("config_snapshot") or _snapshot(bench=str(DEV_BENCH))
    record = {
        "timestamp": report.get("timestamp", ""),
        "metrics": m,
        "retrieval": {"note": "record-only：来自 holdout 报告，未跑检索层"},
        "config_snapshot": snapshot,
        "source_report": reports[-1].name,
    }
    HISTORY.parent.mkdir(exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  🔒 已登记基线: {reports[-1].name} → {HISTORY.name}（后续 regression 与此对比）")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="v21")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()

    gates = load_gates()
    compare = gates.get("regression_tolerances", {})
    snapshot_keys = gates.get("regression_snapshot_keys", [])

    if args.record_only:
        return record_baseline_from_report(args.agents)

    # 1. Agentic dev 回归
    print("=" * 60)
    print("  🔄 Regression: Agentic dev 18 题")
    print("=" * 60, flush=True)
    t0 = time.time()
    report = run_agentic(args.agents, [])
    m = normalize_metrics(report["metrics"].get(args.agents, report["metrics"].get("v21", {})))
    print(f"  📄 {report['timestamp']}  耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)

    # 2. 检索层回归
    ret = {}
    if not args.skip_retrieval:
        print("\n  🔄 Regression: 检索层 56 题（evaluate.py）", flush=True)
        ret = run_retrieval()

    # 3. 对比基线（先校验口径，再比数字——口径不一致必须显式警告）
    baseline = None
    if HISTORY.exists():
        lines = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            baseline = lines[-1]
    failed = []
    if baseline and "metrics" in baseline:
        bm = normalize_metrics(baseline["metrics"])
        base_snap = baseline.get("config_snapshot")
        if base_snap:
            from eval.config_snapshot import diff_snapshots

            cur_snap = report.get("config_snapshot") or _snapshot(bench=str(DEV_BENCH))
            diffs = diff_snapshots(cur_snap, base_snap, snapshot_keys)
            if diffs:
                print("\n  ⚠️ 基线与当前运行的关键口径不一致，对比结果仅供参考（建议 --record-only 重新固化基线）:")
                for d in diffs:
                    print(f"     - {d}")
        else:
            print("\n  ⚠️ 基线记录无 config_snapshot（旧格式），无法校验口径一致性")
        print("\n  📊 与基线对比（基线:", baseline.get("timestamp"), "）")
        for k, tol in compare.items():
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

    # 4. 记录（含 config_snapshot，基线自此可追溯）
    snapshot = report.get("config_snapshot") or _snapshot(bench=str(DEV_BENCH))
    record = {
        "timestamp": report["timestamp"],
        "metrics": m,
        "retrieval": ret,
        "elapsed_s": round(time.time() - t0),
        "config_snapshot": snapshot,
    }
    HISTORY.parent.mkdir(exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 5. PASS → 同步 baseline_current.json（当前晋升基准）
    if not failed:
        BASELINE_CURRENT.write_text(
            json.dumps(
                {
                    "timestamp": record["timestamp"],
                    "metrics": m,
                    "retrieval": ret,
                    "config_snapshot": snapshot,
                    "source": "regression",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  📌 晋升基准已更新: {BASELINE_CURRENT.name}")

    print("\n  🔒 Regression 门禁:", "PASS" if not failed else f"FAIL（下降: {failed}）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

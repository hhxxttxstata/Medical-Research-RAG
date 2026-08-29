"""
qualify.py — Candidate 资格门禁（bad case → candidate → **qualification**）

两级模式：
  --rules  纯规则断言（无需索引/LLM，CI 可跑）：
           - 结构信号判定（_is_multi_part / _is_comparison）与 probe 期望一致
           - 已登记坏例的规则断言（bad_cases.json 的 code-side 对应）
  --full   真实评测门禁（需重建索引 + DeepSeek API，串行独占）：
           - 跑 bad case 对应的 dev 题子集（v2.1）
           - Exit Criteria 5 条断言（Rescue / Harm / OOD / FalseAbstain / Unnecessary）

用法:
    python scripts/qualify.py --rules                 # 快速规则门禁
    python scripts/qualify.py --full                  # 真实评测门禁（20-40 分钟）
    python scripts/qualify.py                         # rules + full

约定（评测纪律 2.0）：
    任何 candidate 必须 qualify 全过 → regression 不降 → 才允许冻结（promotion）。
    新发现 bad case 先登记 tests/bad_cases.json，再补规则断言于此。
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
PROBES = ROOT / "tests" / "policy_probes.json"
BAD_CASES = ROOT / "tests" / "bad_cases.json"


# ══════════════════════════════════════════════════════
#  --rules：纯规则断言（CI 可跑，无需索引/LLM）
# ══════════════════════════════════════════════════════


def _rules_checks() -> list[tuple[str, bool, str]]:
    """返回 [(case_name, passed, detail)]——每个已登记坏例的代码侧断言"""
    from src.agentic_rag import AgenticRAG

    is_mp = AgenticRAG._is_multi_part
    is_cmp = AgenticRAG._is_comparison
    checks = []

    # bh_easy_03_hard_02_03（B1）：对比/并列两级化
    checks.append(
        (
            "bh_easy: 窗宽和窗位的区别 → 非 multi-part（移入 comparison）",
            is_mp("窗宽和窗位的区别是什么？") is False and is_cmp("窗宽和窗位的区别是什么？") is True,
            f"mp={is_mp('窗宽和窗位的区别是什么？')} cmp={is_cmp('窗宽和窗位的区别是什么？')}",
        )
    )
    checks.append(
        (
            "bh_multi_01: 双问号 → multi-part（可靠结构信号）",
            is_mp("肺栓塞的诊断标准是什么？溶栓治疗的适应症有哪些？") is True,
            f"mp={is_mp('肺栓塞的诊断标准是什么？溶栓治疗的适应症有哪些？')}",
        )
    )
    checks.append(
        (
            "追问（共享实体）不算 multi-part",
            is_mp("DICOM是什么？转换公式是什么？") is False,
            f"mp={is_mp('DICOM是什么？转换公式是什么？')}",
        )
    )
    checks.append(
        (
            "极短裸问不算 multi-part",
            is_mp("内部测试集的AUC是多少？外部验证集呢？") is False,
            f"mp={is_mp('内部测试集的AUC是多少？外部验证集呢？')}",
        )
    )

    # policy_probes.json：accept 类 probe 不应被结构信号判为拆解
    probes = json.loads(PROBES.read_text(encoding="utf-8")).get("probes", [])
    for p in probes:
        q = p["question"]
        cat = p.get("category", "")
        if cat == "accept":
            ok = is_mp(q) is False and is_cmp(q) is False
            checks.append((f"probe {p['id']} [accept] 不被误判拆解", ok, f"mp={is_mp(q)} cmp={is_cmp(q)}"))
        elif cat == "decompose":
            ok = is_mp(q) or is_cmp(q)
            checks.append((f"probe {p['id']} [decompose] 触发结构信号", ok, f"mp={is_mp(q)} cmp={is_cmp(q)}"))

    return checks


def run_rules() -> int:
    print("=" * 60)
    print("  🔍 Qualify --rules（纯规则门禁，无需索引/LLM）")
    print("=" * 60)
    checks = _rules_checks()
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name}  [{detail}]")
        failed += not ok
    print(f"\n  规则断言: {len(checks) - failed}/{len(checks)} 通过")
    return 1 if failed else 0


# ══════════════════════════════════════════════════════
#  --full：真实评测门禁（bad case 子集 + Exit Criteria）
# ══════════════════════════════════════════════════════

EXIT_CRITERIA = [
    # (name, 断言函数 metrics -> bool, 说明)
]


def run_full(only_ids: list[str], report_path: str = "") -> int:
    print("=" * 60)
    print("  🔬 Qualify --full（真实评测门禁：bad case dev 子集 + Exit Criteria）")
    print("=" * 60, flush=True)

    if not report_path:
        if not only_ids:
            # 默认：坏例登记的 dev case + ood 全量（仅保留 dev 题集内存在的 id）
            bc = json.loads(BAD_CASES.read_text(encoding="utf-8"))["bad_cases"]
            bench_ids = {b["id"] for b in json.loads(DEV_BENCH.read_text(encoding="utf-8")).get("benchmark", [])}
            only_ids = []
            for c in bc:
                for tok in re.findall(r"[A-Za-z0-9_]+", c.get("dev_case", "") or ""):
                    if tok in bench_ids:
                        only_ids.append(tok)
            only_ids += ["bh_ood_01", "bh_ood_02"]
            only_ids = list(dict.fromkeys(only_ids))
            print(f"  🎯 bad case 子集（dev 题集内）: {only_ids}", flush=True)

        cmd = [
            sys.executable,
            "-u",
            "scripts/holdout_eval.py",
            "--agents",
            "v21",
            "--bench",
            str(DEV_BENCH),
            "--only",
            ",".join(only_ids),
        ]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        # 读最新报告
        reports = sorted((ROOT / "eval_results").glob("holdout30_*.json"), key=lambda p: p.stat().st_mtime)
        if not reports:
            print("  ❌ 未找到评测报告")
            return 1
        report_path = str(reports[-1])
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    m = report["metrics"]["v21"]
    # holdout_eval 报告存显示名（"Harm" 等），映射回字段名
    KEY_MAP = {
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
    m = {k: m.get(v) for k, v in KEY_MAP.items()}
    # 可答题 ER（排除 OOD——无 gold 计 0 会把分母拉低）
    details = report.get("details", {}).get("v21", [])
    ans_det = [d for d in details if d.get("type") != "unsupported_ood"]
    er_answerable = sum(d.get("v1_evidence_recall", 0.0) for d in ans_det) / max(len(ans_det), 1) if ans_det else 0.0
    print(f"\n  📊 bad case 子集结果（{Path(report_path).name}）:")
    for k in (
        "final_answer_accuracy",
        "evidence_recall",
        "hop_recall",
        "completeness",
        "final_rescue",
        "harm",
        "ood_reject",
        "false_abstain",
        "policy_action_accuracy",
        "decomposition_success",
        "unnecessary_action_rate",
    ):
        print(f"    {k:<28}{m.get(k)}")
    print(f"    {'er_answerable（可答题）':<28}{er_answerable:.3f}")

    # Exit Criteria 断言
    failed = 0

    def crit(name: str, ok: bool, detail: str = ""):
        nonlocal failed
        print(f"  {'✅' if ok else '❌'} Exit: {name}  {detail}")
        failed += not ok

    crit("Harm = 0", int(m.get("harm", 0)) == 0, f"harm={m.get('harm')}")
    crit(
        "OOD 全部拒答",
        m.get("ood_reject", "0/0").split("/")[0] == m.get("ood_reject", "0/0").split("/")[1],
        f"ood_reject={m.get('ood_reject')}",
    )
    crit(
        "False Abstain = 0",
        m.get("false_abstain", "1/1").split("/")[0] == "0",
        f"false_abstain={m.get('false_abstain')}",
    )
    crit(
        "Unnecessary Action = 0",
        m.get("unnecessary_action_rate", "1/1").split("/")[0] == "0",
        f"unnecessary={m.get('unnecessary_action_rate')}",
    )
    crit("Evidence Recall ≥ 0.8（可答题口径）", er_answerable >= 0.8, f"ER(可答题)={er_answerable:.3f}")
    crit(
        "Decomp 分支活跃（bad case 含多跳题）",
        int(m.get("decomposition_success", 0)) > 0,
        f"decomp={m.get('decomposition_success')}",
    )

    print(f"\n  耗时 {report.get('elapsed', '?')}s")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", action="store_true", help="只跑纯规则门禁")
    parser.add_argument("--full", action="store_true", help="只跑真实评测门禁")
    parser.add_argument("--report", default="", help="用已有评测报告离线判定（跳过评测）")
    parser.add_argument("--only", default="", help="bad case dev 题 id 子集（--full 时）")
    args = parser.parse_args()

    rc = 0
    if not args.full:
        rc |= run_rules()
    if not args.rules:
        only = [x.strip() for x in args.only.split(",") if x.strip()]
        rc |= run_full(only, report_path=args.report)
    print("\n  🔒 Qualify 门禁:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())

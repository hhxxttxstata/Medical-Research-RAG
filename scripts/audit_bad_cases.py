"""Bad case 登记表审计（issue #4 七阶段闭环 ①收集）

校验 tests/bad_cases.json 的 schema 与登记纪律，任何新 case 必须"先登记再修复"：
  - 必填字段：id / discovered / problem / expected_behavior / status
  - dev_case 仅当缺陷可映射到单个 dev 题时填写；聚合型/指标型缺陷可为空
  - status 限于五态流转，非 closed 的 case 不得晋升（qualify/regression 门禁兜底）
  - closed 是终态：必须写清 root_cause / fix / verification（可追溯）
  - localize（②定位产物，由 scripts/locate_failure.py 写入）若存在，surface 须合法

用法:
    python scripts/audit_bad_cases.py                # 默认校验 tests/bad_cases.json
    python scripts/audit_bad_cases.py --path <file>  # 指定路径（测试/其他登记表）

退出码: 0 = 全部合规；1 = 存在违规（打印逐条清单）
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "tests" / "bad_cases.json"

# 五态流转：open → investigating → fixing → verifying → closed（只进不退）
STATUS_VOCABULARY = {"open", "investigating", "fixing", "verifying", "closed"}

# localize.surface 词表（与 scripts/locate_failure.py 的四类失败面对应）
LOCALIZE_SURFACES = {"retrieval", "policy", "generation", "refusal"}

REQUIRED_FIELDS = ("id", "discovered", "problem", "expected_behavior", "status")
CLOSED_REQUIRED = ("root_cause", "fix", "verification")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def audit(path: Path) -> list[str]:
    """返回违规清单；空列表 = 全部合规。纯函数，便于单元测试。"""
    violations: list[str] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"文件不存在: {path}"]
    except json.JSONDecodeError as e:
        return [f"JSON 解析失败: {e}"]

    cases = data.get("bad_cases")
    if not isinstance(cases, list):
        return ["顶层缺少 bad_cases 数组"]

    seen_ids: set[str] = set()
    for i, case in enumerate(cases):
        cid = case.get("id") or f"<第 {i + 1} 条缺 id>"

        for field in REQUIRED_FIELDS:
            if not case.get(field):
                violations.append(f"{cid}: 缺必填字段 {field}")

        status = case.get("status")
        if status and status not in STATUS_VOCABULARY:
            violations.append(f"{cid}: 非法 status '{status}'（允许: {sorted(STATUS_VOCABULARY)}）")

        discovered = case.get("discovered")
        if discovered and not DATE_RE.match(str(discovered)):
            violations.append(f"{cid}: discovered '{discovered}' 不是 YYYY-MM-DD")

        if cid in seen_ids:
            violations.append(f"{cid}: id 重复")
        seen_ids.add(cid)

        if status == "closed":
            for field in CLOSED_REQUIRED:
                if not case.get(field):
                    violations.append(f"{cid}: closed 但缺 {field}（终态必须可追溯）")

        localize = case.get("localize")
        if localize is not None and (
            not isinstance(localize, dict) or localize.get("surface") not in LOCALIZE_SURFACES
        ):
            violations.append(f"{cid}: localize 格式非法（应为 {{surface: {sorted(LOCALIZE_SURFACES)}, detail: str}}）")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="bad case 登记表 schema 审计")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="登记表路径")
    args = parser.parse_args()

    violations = audit(args.path)
    if violations:
        print(f"❌ bad case 登记表审计未通过（{len(violations)} 项违规）:")
        for v in violations:
            print(f"  - {v}")
        return 1
    n = len(json.loads(args.path.read_text(encoding="utf-8"))["bad_cases"])
    print(f"✅ bad case 登记表审计通过（{n} 条 case 全部合规）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""失败面定位（issue #4 七阶段闭环 ②定位）

对评测报告中失败的题目自动归类失败面，结论写回 tests/bad_cases.json 的
localize 字段，供 ③根因 人工填写 root_cause 前聚焦。

四类失败面（判定依据 = 报告 details 逐题记录 + benchmark 题目元数据）:
  - refusal   误拒/漏拒：abstained 与 expected_route 终局动作不符
  - policy    路由错误：route 与 expected_route 不符（policy_action_accuracy 口径）
  - retrieval gold 缺失：Evidence Recall < 1.0（hop 级 gold 不在 final evidence）
  - generation 证据齐但答案错：Evidence Recall = 1.0 且 Final Answer Acc = false

优先级（主 surface）: refusal > policy > retrieval > generation；
一次失败可命中多个面（all_surfaces 全量保留）。

用法:
    python scripts/locate_failure.py --case bh_multi_01            # 定位单条 case
    python scripts/locate_failure.py                               # 最新报告全部失败题
    python scripts/locate_failure.py --report <path> --case <id>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.failure_taxonomy import suggest_for_surface  # noqa: E402
from eval.rescue_metrics import policy_action_accuracy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BAD_CASES = ROOT / "tests" / "bad_cases.json"
REPORT_DIR = ROOT / "eval_results"

SURFACES = ("refusal", "policy", "retrieval", "generation")  # 主 surface 优先级
AGENT = "v21"


def classify(detail: dict, case_item: dict) -> dict:
    """单题失败面分类（纯函数）。

    detail: 报告 details[agent] 中的逐题记录
    case_item: benchmark 题目元数据（含 expected_route / hops）
    返回: {"surfaces": [...按优先级...], "suggested_taxonomy": [...建议根因码...], "summary": str}
    """
    surfaces: list[str] = []
    notes: list[str] = []

    expected_route = case_item.get("expected_route") or []
    route = detail.get("route") or []
    abstained = bool(detail.get("abstained"))

    # ① refusal：终局动作与预期不符（漏拒 = 该拒没拒；误拒 = 不该拒拒了）
    # 约定与 rescue_metrics 一致：expected_route 为空 ⇒ 期望 ABSTAIN
    expected_abstain = (not expected_route) or expected_route[-1] == "ABSTAIN"
    if abstained != expected_abstain:
        surfaces.append("refusal")
        kind = "误拒" if abstained else "漏拒"
        notes.append(f"{kind}: abstained={abstained}, expected={'ABSTAIN' if expected_abstain else 'ACCEPT'}")

    # ② policy：终局动作一致但循环内动作不符（终局不匹配已归入 refusal）
    if expected_route and route[-1] == expected_route[-1] and not policy_action_accuracy(route, expected_route):
        surfaces.append("policy")
        notes.append(f"route={route} != expected={expected_route}")

    # ③④ retrieval / generation：仅对实际作答的题有意义
    # （正确拒答的 OOD 题 gold 本就不存在，ER=0 是正常现象，不算失败面）
    if abstained:
        return _result(surfaces, notes)

    er = float(detail.get("v1_evidence_recall", 1.0))
    n_hops = len(case_item.get("hops") or [])
    if er < 1.0:
        surfaces.append("retrieval")
        notes.append(f"Evidence Recall={er:.2f}（hop 数 {n_hops}，gold 未全部命中 final evidence）")

    fa_ok = detail.get("final_answer_accuracy")
    if fa_ok is False and not surfaces:
        surfaces.append("generation")
        notes.append("证据已齐（Recall=1.0）但 Final Answer Acc=false")

    return _result(surfaces, notes)


def _result(surfaces: list[str], notes: list[str]) -> dict:
    """统一组装 classify 结果（主失败面 → 建议根因码，供 ③根因 参考）"""
    return {
        "surfaces": surfaces,
        "suggested_taxonomy": suggest_for_surface(surfaces[0]) if surfaces else [],
        "summary": "；".join(notes) or "无失败面（本题通过）",
    }


def locate(case_ids: list[str], report_path: Path) -> dict[str, dict]:
    """在报告中定位指定题目（dev_case id → benchmark 题目 id），返回 {id: classify 结果}"""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details = {d["id"]: d for d in report["details"][AGENT]}
    items = {b["id"]: b for b in report["cases"] for b in [b["question"]]}

    out: dict[str, dict] = {}
    for cid in case_ids:
        if cid not in details:
            out[cid] = {"surfaces": [], "summary": f"报告 {report_path.name} 中无题目 {cid}"}
            continue
        out[cid] = classify(details[cid], items.get(cid, {}))
    return out


def latest_report() -> Path:
    files = sorted(REPORT_DIR.glob("holdout30_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"❌ {REPORT_DIR} 下无 holdout30_*.json 报告，先跑 scripts/holdout_eval.py")
    return files[-1]


def write_localize(case_ids: list[str], results: dict[str, dict]) -> int:
    """把定位结论写回 bad_cases.json（按 dev_case 匹配），返回更新条数"""
    data = json.loads(BAD_CASES.read_text(encoding="utf-8"))
    updated = 0
    for case in data["bad_cases"]:
        dev = case.get("dev_case")
        if dev in results:
            r = results[dev]
            if not r["surfaces"]:
                continue  # 本题通过，无需定位
            case["localize"] = {
                "surface": r["surfaces"][0],
                "all_surfaces": r["surfaces"],
                "suggested_taxonomy": r.get("suggested_taxonomy", []),
                "detail": r["summary"],
            }
            updated += 1
    if updated:
        BAD_CASES.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="失败面定位（四类：retrieval/policy/generation/refusal）")
    parser.add_argument("--case", action="append", default=[], help="dev_case/题目 id，可多次指定；缺省=报告全部失败题")
    parser.add_argument("--report", type=Path, default=None, help="评测报告路径（默认最新 holdout30_*.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印结论，不写回 bad_cases.json")
    args = parser.parse_args()

    report_path = args.report or latest_report()
    print(f"📄 报告: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "cases" not in report:
        raise SystemExit(f"❌ 报告 {report_path.name} 缺 cases 字段（旧格式，无逐题元数据），请改用含 cases 的新报告")
    items = {b["question"]["id"]: b["question"] for b in report["cases"]}
    if args.case:
        case_ids = args.case
    else:
        all_details = report["details"][AGENT]
        failed = [d["id"] for d in all_details if _failed(d, items.get(d["id"], {}))]
        case_ids = failed
        print(f"🔍 报告失败题: {case_ids or '无'}")

    results = locate(case_ids, report_path)
    for cid, r in results.items():
        sugg = ",".join(r.get("suggested_taxonomy", [])[:3]) or "-"
        print(f"  {cid:>24}: {','.join(r['surfaces']) or '-'} | 建议码: {sugg} | {r['summary']}")

    if args.dry_run:
        print("（dry-run，未写回）")
        return 0

    updated = write_localize(case_ids, results)
    print(f"✍️  已写回 {BAD_CASES.name}: {updated} 条 localize 字段")
    return 0


def _failed(detail: dict, case_item: dict) -> bool:
    """题目是否失败：误拒/漏拒、路由错、gold 缺失、答案错 任一即失败"""
    r = classify(detail, case_item)
    return bool(r["surfaces"])


if __name__ == "__main__":
    sys.exit(main())

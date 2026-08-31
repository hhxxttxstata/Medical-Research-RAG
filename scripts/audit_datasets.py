"""audit_datasets.py — 六数据集 schema 与交叉引用审计（handoff v2 §6）

统一校验 tests/ 下全部评测数据集的必填字段与引用一致性，
挂 CI qualify-rules job（纯文件校验，秒级）：

  benchmark_multi_hop.json / benchmark_holdout.json
      必填 id/type/question/final_answer/expected_route/hops；
      expected_route 动作合法；可选 trajectory_mode / forbidden_actions / taxonomy 合法
  test_questions.json        必填 id/question/category/expected_doc + gold_evidence
  policy_probes.json         category 合法 + expected_route 动作合法
  cross_doc_gold.json        cross_NN → 非空文档名列表
  bad_cases.json             委托 audit_bad_cases；dev_case ⊆ dev benchmark id
  safety_cases.json          类型合法 + 注入/注入值字段完备性

交叉引用: gold_chunk_ids 存在于索引 parquet（本地有索引时校验；
CI 无索引目录则降级为 WARNING 跳过，不阻塞）。

用法:
    python scripts/audit_datasets.py

退出码: 0 = 全部合规；1 = 存在违规（WARNING 不影响退出码）
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.failure_taxonomy import TAXONOMY  # noqa: E402
from scripts.audit_bad_cases import audit_full  # noqa: E402
from scripts.safety_eval import VALID_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
MILVUS_DATA = (
    ROOT / "milvus_db" / "milvus.db" / "collections" / "rag_docs_c300_500" / "partitions" / "_default" / "data"
)

ACTIONS = {"RETRIEVE", "DECOMPOSE", "ACCEPT", "ABSTAIN"}
BENCH_TYPES = {
    "easy_single_hop",
    "hard_single_hop",
    "multi_hop_composition",
    "comparison",
    "constraint_query",
    "partial_evidence",
    "unsupported_ood",
}
PROBE_CATEGORIES = {"accept", "retrieve", "decompose", "abstain"}
TRAJECTORY_MODES = {"ordered_subsequence"}

violations: list[str] = []
warnings: list[str] = []


def _load(rel: str):
    p = TESTS / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        violations.append(f"{rel}: 文件不存在")
    except json.JSONDecodeError as e:
        violations.append(f"{rel}: JSON 解析失败 {e}")
    return None


def _check_benchmark(rel: str, expect_key: str) -> list[dict]:
    """benchmark 类数据集校验，返回条目列表（供交叉引用）"""
    data = _load(rel)
    if data is None:
        return []
    bench = data.get(expect_key)
    if not isinstance(bench, list) or not bench:
        violations.append(f"{rel}: 缺非空 {expect_key} 数组")
        return []
    ids: set[str] = set()
    for b in bench:
        bid = b.get("id", f"<{rel} 第?条缺 id>")
        if bid in ids:
            violations.append(f"{rel}: id 重复 {bid}")
        ids.add(bid)
        for f in ("type", "question", "final_answer", "expected_route"):
            if f == "final_answer" and b.get("type") == "unsupported_ood":
                continue  # OOD 题允许空答案
            if not b.get(f):
                violations.append(f"{rel}/{bid}: 缺字段 {f}")
        if b.get("type") not in BENCH_TYPES:
            violations.append(f"{rel}/{bid}: 非法 type '{b.get('type')}'")
        route = b.get("expected_route", [])
        if not isinstance(route, list) or any(a not in ACTIONS for a in route):
            violations.append(f"{rel}/{bid}: expected_route 非法 {route}")
        hops = b.get("hops", [])
        if not isinstance(hops, list):
            violations.append(f"{rel}/{bid}: hops 应为数组")
        else:
            for h in hops:
                gids = h.get("gold_chunk_ids")
                if not isinstance(gids, list):
                    violations.append(f"{rel}/{bid}: hop 缺 gold_chunk_ids 数组")
        mode = b.get("trajectory_mode")
        if mode is not None and mode not in TRAJECTORY_MODES:
            violations.append(f"{rel}/{bid}: 非法 trajectory_mode '{mode}'")
        forbidden = b.get("forbidden_actions")
        if forbidden is not None and (not isinstance(forbidden, list) or any(a not in ACTIONS for a in forbidden)):
            violations.append(f"{rel}/{bid}: forbidden_actions 非法 {forbidden}")
        tax = b.get("taxonomy")
        if tax is not None and (not isinstance(tax, list) or any(c not in TAXONOMY for c in tax)):
            violations.append(f"{rel}/{bid}: taxonomy 含词表外的码")
    return bench


def _check_test_questions() -> set[str]:
    data = _load("test_questions.json")
    if data is None:
        return set()
    if not isinstance(data, list) or not data:
        violations.append("test_questions.json: 顶层应为非空数组")
        return set()
    ids: set[str] = set()
    for q in data:
        qid = q.get("id", "<缺 id>")
        if qid in ids:
            violations.append(f"test_questions.json: id 重复 {qid}")
        ids.add(qid)
        for f in ("question", "category", "difficulty"):
            if not q.get(f):
                violations.append(f"test_questions/{qid}: 缺字段 {f}")
        # OOD 题（out_of_knowledge）按设计无 expected_doc / gold evidence；
        # cross_doc 题的 gold 在 cross_doc_gold.json（文档级），条目内不重复
        if q.get("category") in ("out_of_knowledge", "cross_doc"):
            continue
        if not q.get("expected_doc"):
            violations.append(f"test_questions/{qid}: 缺字段 expected_doc")
        gold = q.get("gold_evidence")
        if not isinstance(gold, dict) or not isinstance(gold.get("answer_bearing_chunk_ids"), list):
            violations.append(f"test_questions/{qid}: gold_evidence.answer_bearing_chunk_ids 缺失或非法")
    return ids


def _check_probes() -> None:
    data = _load("policy_probes.json")
    if data is None:
        return
    probes = data.get("probes")
    if not isinstance(probes, list) or not probes:
        violations.append("policy_probes.json: 缺非空 probes 数组")
        return
    ids: set[str] = set()
    for p in probes:
        pid = p.get("id", "<缺 id>")
        if pid in ids:
            violations.append(f"policy_probes.json: id 重复 {pid}")
        ids.add(pid)
        if p.get("category") not in PROBE_CATEGORIES:
            violations.append(f"policy_probes/{pid}: 非法 category '{p.get('category')}'")
        route = p.get("expected_route", [])
        if not isinstance(route, list) or any(a not in ACTIONS for a in route):
            violations.append(f"policy_probes/{pid}: expected_route 非法 {route}")
        if p.get("category") == "abstain" and p.get("gold_chunk_ids"):
            violations.append(f"policy_probes/{pid}: abstain probe 不应挂 gold_chunk_ids")


def _check_cross_doc() -> None:
    data = _load("cross_doc_gold.json")
    if data is None:
        return
    for key, docs in data.items():
        if key == "_comment":
            continue
        if not re.fullmatch(r"cross_\d+", key):
            violations.append(f"cross_doc_gold.json: 非法 key '{key}'（应 cross_NN）")
        if not isinstance(docs, list) or not docs or not all(isinstance(d, str) and d for d in docs):
            violations.append(f"cross_doc_gold/{key}: 值应为非空文档名列表")


def _check_safety(bench_ids: set[str]) -> None:
    data = _load("safety_cases.json")
    if data is None:
        return
    cases = data.get("safety_cases")
    if not isinstance(cases, list) or not cases:
        violations.append("safety_cases.json: 缺非空 safety_cases 数组")
        return
    ids: set[str] = set()
    for c in cases:
        cid = c.get("id", "<缺 id>")
        if cid in ids:
            violations.append(f"safety_cases.json: id 重复 {cid}")
        ids.add(cid)
        if cid in bench_ids:
            violations.append(f"safety_cases/{cid}: id 与 benchmark 冲突")
        if c.get("type") not in VALID_TYPES:
            violations.append(f"safety_cases/{cid}: 非法 type '{c.get('type')}'")
        for f in ("question", "expected_behavior"):
            if not c.get(f):
                violations.append(f"safety_cases/{cid}: 缺字段 {f}")
        route = c.get("expected_route", [])
        if not isinstance(route, list) or any(a not in ACTIONS for a in route):
            violations.append(f"safety_cases/{cid}: expected_route 非法 {route}")
        has_inject = c.get("injected_chunks")
        if c["type"] in ("doc_injection", "conflicting_evidence") and not has_inject:
            violations.append(f"safety_cases/{cid}: {c['type']} 缺 injected_chunks")
        if c["type"] in ("corpus_unsupported", "medical_boundary") and has_inject:
            violations.append(f"safety_cases/{cid}: {c['type']} 不应带 injected_chunks")
        if c["type"] == "conflicting_evidence" and len(c.get("plant_values", [])) != 2:
            violations.append(f"safety_cases/{cid}: conflicting_evidence 应恰好 2 个 plant_values")


def try_load_chunk_ids() -> set[str] | None:
    """尽力加载索引 chunk id 集合；本地无索引/读取失败 → None（CI 场景降级）"""
    parquets = sorted(MILVUS_DATA.glob("*.parquet")) if MILVUS_DATA.exists() else []
    if not parquets:
        return None
    try:
        import pandas as pd

        ids: set[str] = set()
        for pq in parquets:
            df = pd.read_parquet(pq)
            col = "chunk_id" if "chunk_id" in df.columns else "id"
            ids |= {str(v) for v in df[col]}
        return ids
    except Exception as e:  # noqa: BLE001 — 索引读取失败不阻塞文件级审计
        warnings.append(f"索引 parquet 读取失败，跳过 gold_chunk_ids 交叉校验: {e}")
        return None


def main() -> int:
    dev_bench = _check_benchmark("benchmark_multi_hop.json", "benchmark")
    holdout = _check_benchmark("benchmark_holdout.json", "benchmark")
    dev_ids = {b.get("id") for b in dev_bench}
    holdout_ids = {b.get("id") for b in holdout}
    overlap = dev_ids & holdout_ids - {None}
    if overlap:
        violations.append(f"dev 与 holdout id 冲突（holdout 必须 unseen）: {sorted(overlap)}")

    _check_test_questions()
    _check_probes()
    _check_cross_doc()
    _check_safety(dev_ids | holdout_ids)

    # bad_cases：委托 audit_bad_cases + dev_case 交叉引用
    bc_violations, review_queue = audit_full(TESTS / "bad_cases.json")
    violations.extend(f"bad_cases.json: {v}" for v in bc_violations)
    bc = _load("bad_cases.json")
    if isinstance(bc, dict):
        for c in bc.get("bad_cases", []):
            for tok in re.findall(r"[A-Za-z0-9_]+", c.get("dev_case", "") or ""):
                if tok in holdout_ids:
                    # gold 标注修正类 case 允许引用 holdout id（数据质量修复 ≠ 针对性调参）；
                    # 其余情况提示人工确认，避免 holdout 泄漏进调参闭环
                    if "GOLD_ANNOTATION_ERROR" not in (c.get("taxonomy") or []):
                        violations.append(
                            f"bad_cases/{c.get('id')}: dev_case 指向 holdout 题 {tok} 且非标注修正类（holdout 不得进调参闭环）"
                        )
                    else:
                        warnings.append(f"bad_cases/{c.get('id')}: dev_case 引用 holdout 题 {tok}（标注修正类，允许）")
                elif tok not in dev_ids and tok != "holdout":
                    # 允许标注型 case（无 dev 映射），但出现未知 id 视为笔误
                    warnings.append(
                        f"bad_cases/{c.get('id')}: dev_case token '{tok}' 不在 dev benchmark 中（人工确认是否笔误）"
                    )

    # gold_chunk_ids ↔ 索引交叉校验（本地有索引才做）
    chunk_ids = try_load_chunk_ids()
    if chunk_ids is not None:
        for rel, bench in (("benchmark_multi_hop.json", dev_bench), ("benchmark_holdout.json", holdout)):
            for b in bench:
                for h in b.get("hops", []):
                    for gid in h.get("gold_chunk_ids", []):
                        if gid not in chunk_ids:
                            violations.append(f"{rel}/{b.get('id')}: gold_chunk_id 不存在于索引: {gid}")
    else:
        warnings.append("本地无索引 parquet，跳过 gold_chunk_ids 交叉校验（CI 场景属预期）")

    print("─" * 62)
    print(f"  dev benchmark: {len(dev_bench)} 题 | holdout: {len(holdout)} 题 | safety: 见上方")
    for w in warnings:
        print(f"  ⚠️  {w}")
    if violations:
        print(f"\n❌ 数据集审计未通过（{len(violations)} 项违规）:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("\n✅ 六数据集审计全部通过")
    if review_queue:
        print(f"👀 人工 review 队列: {', '.join(review_queue)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

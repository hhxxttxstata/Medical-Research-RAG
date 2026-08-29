"""audit_bad_cases 单元测试（issue #4 ①收集：登记表 schema 校验）"""

import json

import pytest

from scripts.audit_bad_cases import audit


def _write(tmp_path, cases, top=None):
    data = {"_comment": "test", "bad_cases": cases, **(top or {})}
    p = tmp_path / "bad_cases.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _case(**overrides):
    base = {
        "id": "bh_x_01",
        "discovered": "2026-08-10",
        "problem": "问题描述",
        "expected_behavior": "期望行为",
        "root_cause": "根因",
        "fix": "修复",
        "verification": "验证",
        "status": "closed",
        "dev_case": "bh_x_01",
    }
    base.update(overrides)
    return base


class TestAudit:
    def test_valid_closed_case_passes(self, tmp_path):
        assert audit(_write(tmp_path, [_case()])) == []

    def test_open_case_needs_no_root_cause(self, tmp_path):
        case = _case(status="open")
        for f in ("root_cause", "fix", "verification"):
            case.pop(f)
        assert audit(_write(tmp_path, [case])) == []

    def test_missing_required_field(self, tmp_path):
        case = _case()
        case.pop("expected_behavior")
        violations = audit(_write(tmp_path, [case]))
        assert any("expected_behavior" in v for v in violations)

    def test_duplicate_id(self, tmp_path):
        violations = audit(_write(tmp_path, [_case(), _case()]))
        assert any("重复" in v for v in violations)

    @pytest.mark.parametrize("bad", ["done", "Closed", "closed "])
    def test_illegal_status(self, tmp_path, bad):
        violations = audit(_write(tmp_path, [_case(status=bad)]))
        assert any("非法 status" in v for v in violations)

    def test_bad_date_format(self, tmp_path):
        violations = audit(_write(tmp_path, [_case(discovered="2026/08/10")]))
        assert any("YYYY-MM-DD" in v for v in violations)

    def test_closed_without_traceability(self, tmp_path):
        case = _case(root_cause="")
        violations = audit(_write(tmp_path, [case]))
        assert any("root_cause" in v for v in violations)

    def test_valid_localize_passes(self, tmp_path):
        case = _case(localize={"surface": "retrieval", "detail": "gold 不在 final evidence"})
        assert audit(_write(tmp_path, [case])) == []

    def test_illegal_localize_surface(self, tmp_path):
        case = _case(localize={"surface": "llm"})
        violations = audit(_write(tmp_path, [case]))
        assert any("localize" in v for v in violations)

    def test_missing_file(self, tmp_path):
        violations = audit(tmp_path / "nope.json")
        assert violations == [f"文件不存在: {tmp_path / 'nope.json'}"]

    def test_broken_json(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        assert any("JSON 解析失败" in v for v in audit(p))

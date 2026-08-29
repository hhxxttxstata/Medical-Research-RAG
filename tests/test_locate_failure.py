"""locate_failure 单元测试（issue #4 ②定位：四类失败面分类）"""

import json
from pathlib import Path

from scripts.locate_failure import classify, write_localize


def _detail(**overrides):
    """一条通过的 v2.1 逐题记录（各面全部健康）"""
    base = {
        "id": "bh_x_01",
        "type": "multi_hop",
        "route": ["RETRIEVE", "DECOMPOSE", "ACCEPT"],
        "abstained": False,
        "v1_evidence_recall": 1.0,
        "final_answer_accuracy": True,
    }
    base.update(overrides)
    return base


def _case_item(**overrides):
    base = {
        "id": "bh_x_01",
        "type": "multi_hop",
        "expected_route": ["RETRIEVE", "DECOMPOSE", "ACCEPT"],
        "hops": [{"hop": 1, "gold_chunk_ids": ["a_1"]}, {"hop": 2, "gold_chunk_ids": ["b_2"]}],
    }
    base.update(overrides)
    return base


def _write_registry(tmp_path, cases):
    p = tmp_path / "bad_cases.json"
    p.write_text(json.dumps({"_comment": "t", "bad_cases": cases}, ensure_ascii=False), encoding="utf-8")
    return p


class TestClassify:
    def test_healthy_case_no_surface(self):
        r = classify(_detail(), _case_item())
        assert r["surfaces"] == []

    def test_refusal_miss_abstain(self):
        """漏拒：expected ABSTAIN 但实际 ACCEPT"""
        d = _detail(route=["RETRIEVE", "ACCEPT"], abstained=False)
        c = _case_item(type="unsupported_ood", expected_route=["RETRIEVE", "ABSTAIN"])
        r = classify(d, c)
        assert r["surfaces"][0] == "refusal"
        assert "漏拒" in r["summary"]

    def test_refusal_false_abstain(self):
        """误拒：expected ACCEPT 但实际 ABSTAIN"""
        d = _detail(route=["RETRIEVE", "ABSTAIN"], abstained=True)
        r = classify(d, _case_item())
        assert r["surfaces"][0] == "refusal"

    def test_policy_route_mismatch(self):
        """路由错：easy 题循环内出现了多余动作"""
        d = _detail(route=["RETRIEVE", "DECOMPOSE", "ACCEPT"])
        c = _case_item(type="easy_single_hop", expected_route=["RETRIEVE", "ACCEPT"])
        r = classify(d, c)
        assert r["surfaces"][0] == "policy"

    def test_retrieval_gold_missing(self):
        """gold 缺失：Evidence Recall < 1.0"""
        d = _detail(v1_evidence_recall=0.5)
        r = classify(d, _case_item())
        assert "retrieval" in r["surfaces"]
        assert "0.50" in r["summary"]

    def test_generation_evidence_ok_answer_wrong(self):
        """generation：证据齐但答案错（且无其他失败面时为主 surface）"""
        d = _detail(final_answer_accuracy=False)
        r = classify(d, _case_item())
        assert r["surfaces"] == ["generation"]

    def test_multi_surface_priority(self):
        """多面命中：refusal > policy > retrieval；证据缺失时 generation 不单列"""
        d = _detail(
            route=["RETRIEVE", "ACCEPT"],
            abstained=False,
            v1_evidence_recall=0.5,
            final_answer_accuracy=False,
        )
        c = _case_item(expected_route=["RETRIEVE", "ABSTAIN"])
        r = classify(d, c)
        assert r["surfaces"] == ["refusal", "retrieval"]

    def test_ood_type_without_expected_route(self):
        """无 expected_route 的 unsupported_ood：abstained=False 即漏拒"""
        d = _detail(route=["RETRIEVE", "ACCEPT"], abstained=False)
        c = _case_item(type="unsupported_ood", expected_route=[], hops=[])
        r = classify(d, c)
        assert "refusal" in r["surfaces"]

    def test_ood_correctly_refused_is_healthy(self):
        """正确拒答的 OOD 题：ER=0 是正常现象，不算失败面"""
        d = _detail(route=["RETRIEVE", "ABSTAIN"], abstained=True, v1_evidence_recall=0.0)
        c = _case_item(type="unsupported_ood", expected_route=["RETRIEVE", "ABSTAIN"], hops=[])
        assert classify(d, c)["surfaces"] == []


class TestWriteLocalize:
    def test_writes_matching_dev_case(self, tmp_path, monkeypatch):
        p = _write_registry(
            tmp_path,
            [
                {"id": "c1", "status": "investigating", "dev_case": "bh_x_01"},
                {"id": "c2", "status": "closed", "dev_case": "bh_y_99"},
            ],
        )
        monkeypatch.setattr("scripts.locate_failure.BAD_CASES", Path(p))

        n = write_localize(
            ["bh_x_01"],
            {
                "bh_x_01": {
                    "surfaces": ["retrieval", "generation"],
                    "summary": "Evidence Recall=0.50",
                }
            },
        )
        assert n == 1
        out = json.loads(p.read_text(encoding="utf-8"))
        loc = out["bad_cases"][0]["localize"]
        assert loc["surface"] == "retrieval"
        assert loc["all_surfaces"] == ["retrieval", "generation"]
        assert out["bad_cases"][1].get("localize") is None

    def test_passed_case_not_written(self, tmp_path, monkeypatch):
        p = _write_registry(tmp_path, [{"id": "c1", "status": "open", "dev_case": "bh_x_01"}])
        monkeypatch.setattr("scripts.locate_failure.BAD_CASES", Path(p))

        n = write_localize(["bh_x_01"], {"bh_x_01": {"surfaces": [], "summary": "无失败面"}})
        assert n == 0
        out = json.loads(p.read_text(encoding="utf-8"))
        assert out["bad_cases"][0].get("localize") is None

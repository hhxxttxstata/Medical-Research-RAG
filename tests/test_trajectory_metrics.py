"""
Trajectory / Decision 轻量指标单元测试（P0-5 / P1-3）

覆盖 eval/rescue_metrics.py 新增确定性指标：
  - required_action_recall / forbidden_action_violation
  - false_accept / premature_accept / per_action_pr
  - compute_agent_capability_metrics 的聚合口径（缺省字段不进分母）
  - holdout_eval.summarize_reliability（observation 聚合，P0-4）
"""

import pytest

from eval.rescue_metrics import (
    compute_agent_capability_metrics,
    false_accept,
    forbidden_action_violation,
    per_action_pr,
    premature_accept,
    required_action_recall,
)
from scripts.holdout_eval import summarize_reliability

# ── required_action_recall ──


def test_rar_all_present():
    assert required_action_recall(["RETRIEVE", "DECOMPOSE", "ACCEPT"], ["RETRIEVE", "DECOMPOSE", "ACCEPT"]) == 1.0


def test_rar_partial():
    assert required_action_recall(["RETRIEVE", "ACCEPT"], ["RETRIEVE", "DECOMPOSE", "ACCEPT"]) == pytest.approx(0.5)


def test_rar_none_required():
    """expected 无必备动作（如直接 ABSTAIN）→ None，不惩罚"""
    assert required_action_recall(["ABSTAIN"], ["ABSTAIN"]) is None
    assert required_action_recall(["RETRIEVE", "ABSTAIN"], ["RETRIEVE", "ABSTAIN"]) == 1.0


# ── forbidden_action_violation ──


def test_forbidden_violation_true():
    """应拒的题却 ACCEPT（ACCEPT 被声明 forbidden）→ True"""
    assert forbidden_action_violation(["RETRIEVE", "ACCEPT"], ["ACCEPT"]) is True


def test_forbidden_clean():
    assert forbidden_action_violation(["RETRIEVE", "ABSTAIN"], ["ACCEPT"]) is False


def test_forbidden_not_declared():
    assert forbidden_action_violation(["RETRIEVE", "ACCEPT"], None) is None
    assert forbidden_action_violation(["RETRIEVE", "ACCEPT"], []) is None


# ── false_accept / premature_accept ──


def test_false_accept_true():
    assert false_accept(["RETRIEVE", "ACCEPT"], ["RETRIEVE", "ABSTAIN"]) is True


def test_false_accept_correct_abstain():
    assert false_accept(["RETRIEVE", "ABSTAIN"], ["RETRIEVE", "ABSTAIN"]) is False


def test_false_accept_no_expected():
    assert false_accept(["ACCEPT"], []) is None


def test_premature_accept_true():
    """期望有 RETRIEVE 却裸 ACCEPT → True"""
    assert premature_accept(["ACCEPT"], ["RETRIEVE", "ACCEPT"]) is True


def test_premature_accept_false_after_retrieval():
    assert premature_accept(["RETRIEVE", "ACCEPT"], ["RETRIEVE", "ACCEPT"]) is False


def test_premature_accept_not_accepted():
    """未 ACCEPT → None（不进分母）"""
    assert premature_accept(["RETRIEVE", "ABSTAIN"], ["RETRIEVE", "ABSTAIN"]) is None


# ── per_action_pr ──


def test_per_action_pr_counts():
    counts = per_action_pr(["RETRIEVE", "ACCEPT"], ["RETRIEVE", "ABSTAIN"])
    assert counts["RETRIEVE"] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts["DECOMPOSE"] == {"tp": 0, "fp": 0, "fn": 0}
    assert counts["ACCEPT"] == {"tp": 0, "fp": 1, "fn": 0}
    assert counts["ABSTAIN"] == {"tp": 0, "fp": 0, "fn": 1}


# ── 聚合口径 ──


def _case(q_overrides: dict, route: list[str]) -> dict:
    q = {
        "id": q_overrides.get("id", "t1"),
        "type": q_overrides.get("type", "easy_single_hop"),
        "hops": [{"hop": 1, "question": "h1", "gold_chunk_ids": ["g1"]}],
        "final_answer": "42",
        "expected_route": q_overrides.get("expected_route", ["RETRIEVE", "ACCEPT"]),
    }
    q.update(q_overrides)
    src = [{"id": "g1", "metadata": {"filename": "f.md"}}]
    return {
        "question": q,
        "v0_sources": src,
        "v1_sources": src,
        "v1_route": route,
        "v1_answer": "42",
        "v1_abstained": False,
    }


def test_aggregate_forbidden_and_rar():
    cases = [
        _case({"id": "a", "forbidden_actions": ["DECOMPOSE"]}, ["RETRIEVE", "ACCEPT"]),
        _case({"id": "b", "forbidden_actions": ["DECOMPOSE"]}, ["DECOMPOSE", "ACCEPT"]),
        _case({"id": "c", "expected_route": ["RETRIEVE", "DECOMPOSE", "ACCEPT"]}, ["RETRIEVE", "ACCEPT"]),
    ]
    m = compute_agent_capability_metrics(cases)
    # forbidden：2 例声明，1 例违规
    assert m["forbidden_action_rate"] == "1/2"
    # required：a 1.0；b expected 缺省含 RETRIEVE 而 route 无 → 0.0；c 0.5 → 均值 0.5
    assert m["required_action_recall"] == pytest.approx(0.5)
    # false_accept：三例 expected 非空均参与判定，无应拒却答 → 0/3
    assert m["false_accept_rate"] == "0/3"
    # premature：三例均 ACCEPT 且 expected 有必备动作，均执行了动作 → 0/3
    assert m["premature_accept_rate"] == "0/3"
    assert m["details"][0]["forbidden_violation"] is False
    assert m["details"][1]["forbidden_violation"] is True
    assert m["details"][0]["trajectory_mode"] == "ordered_subsequence"


def test_aggregate_false_accept_and_premature():
    cases = [
        # 应拒却答 → false accept，同时裸 ACCEPT 无动作 → premature
        _case({"id": "x", "type": "unsupported_ood", "expected_route": ["RETRIEVE", "ABSTAIN"]}, ["ACCEPT"]),
        # 正确拒答
        _case({"id": "y", "type": "unsupported_ood", "expected_route": ["RETRIEVE", "ABSTAIN"]}, ["ABSTAIN"]),
    ]
    m = compute_agent_capability_metrics(cases)
    assert m["false_accept_rate"] == "1/2"
    # premature：仅 x ACCEPT（违规）；y 拒答不进分母
    assert m["premature_accept_rate"] == "1/1"
    # ABSTAIN：仅 y 预测且命中 → P=1.0；期望 2 例 → R=0.5
    assert m["per_action_precision"]["ABSTAIN"] == pytest.approx(1.0)
    assert m["per_action_recall"]["ABSTAIN"] == pytest.approx(0.5)


# ── summarize_reliability ──


def test_summarize_reliability_from_observation():
    cases = [
        {
            "question": {"id": "a"},
            "v21_answer": "ok",
            "v21_observation": {
                "latency_ms": 100,
                "grader_calls": 1,
                "policy_llm_calls": 0,
                "generation_calls": 1,
                "retrieval_calls": 2,
                "fallback_used": False,
                "operational_error": "none",
            },
        },
        {
            "question": {"id": "b"},
            "v21_answer": "[OPERATIONAL_ERROR] timeout",
            "v21_observation": {
                "latency_ms": 300,
                "grader_calls": 0,
                "policy_llm_calls": 1,
                "generation_calls": 0,
                "retrieval_calls": 1,
                "fallback_used": True,
                "operational_error": "timeout",
            },
        },
    ]
    r = summarize_reliability(cases)
    assert r["n"] == 2
    assert r["latency_ms"]["p50"] == 100  # 最近邻：下位取值
    assert r["latency_ms"]["p95"] == 300
    assert r["latency_ms"]["max"] == 300
    assert r["grader_calls_per_query"] == 0.5
    assert r["timeout_count"] == 1
    assert r["fallback_count"] == 1
    assert r["operational_failure_rate"] == "1/2"
    assert r["malformed_output_rate"] == "1/2"


def test_summarize_reliability_empty():
    assert summarize_reliability([{"question": {"id": "a"}}]) == {}

"""
Step 12: Multi-hop / Agent Capability 指标族单元测试

覆盖 eval/rescue_metrics.py 中 Step 12 新增指标：
  - hop_gold_ids / evidence_recall_at_k / hop_recall_at_k / evidence_completeness
  - final_answer_accuracy / policy_action_accuracy / retry_recovery / decomposition_success
  - unnecessary_action_rate / compute_agent_capability_metrics
"""

import pytest

from eval.rescue_metrics import (
    compute_agent_capability_metrics,
    decomposition_success,
    evidence_completeness,
    evidence_recall_at_k,
    final_answer_accuracy,
    hop_gold_ids,
    hop_recall_at_k,
    policy_action_accuracy,
    retry_recovery,
    unnecessary_action_rate,
)

Q_MULTI = {
    "id": "t_multi",
    "type": "multi_hop_composition",
    "hops": [
        {"hop": 1, "question": "h1", "gold_chunk_ids": ["a1", "a2"]},
        {"hop": 2, "question": "h2", "gold_chunk_ids": ["b1"]},
    ],
    "final_answer": "42",
    "expected_route": ["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"],
}


def test_hop_gold_ids():
    golds = hop_gold_ids(Q_MULTI)
    assert golds == [{"a1", "a2"}, {"b1"}]


def test_evidence_recall_at_k():
    src = [{"id": "a1"}, {"id": "x"}, {"id": "b1"}]
    assert evidence_recall_at_k(src, {"a1", "a2", "b1"}) == pytest.approx(2 / 3)
    assert evidence_recall_at_k([{"id": "x"}], {"a1"}) == 0.0
    assert evidence_recall_at_k(src, set()) == 0.0


def test_hop_recall_at_k():
    # 两个 hop 都命中
    src = [{"id": "a1"}, {"id": "b1"}]
    assert hop_recall_at_k(src, Q_MULTI) == 1.0
    # 只命中一个 hop
    assert hop_recall_at_k([{"id": "b1"}], Q_MULTI) == 0.5
    # 一个没命中
    assert hop_recall_at_k([{"id": "x"}], Q_MULTI) == 0.0


def test_evidence_completeness():
    # 全 gold 命中
    src = [{"id": "a1"}, {"id": "a2"}, {"id": "b1"}]
    assert evidence_completeness(src, Q_MULTI) == 1.0
    # 部分
    assert evidence_completeness([{"id": "a1"}], Q_MULTI) == pytest.approx(1 / 3)


def test_final_answer_accuracy():
    assert final_answer_accuracy("答案是42个", "42") is True
    assert final_answer_accuracy("答案是43个", "42") is False
    assert final_answer_accuracy("", "42") is False
    assert final_answer_accuracy("随便说", "") is False


def test_policy_action_accuracy():
    # easy：无循环内动作
    assert policy_action_accuracy(["RETRIEVE", "ACCEPT"], ["RETRIEVE", "ACCEPT"]) is True
    # easy 但过度动作 → False
    assert policy_action_accuracy(["RETRIEVE", "RETRIEVE", "ACCEPT"], ["RETRIEVE", "ACCEPT"]) is False
    # multi-hop：有 DECOMPOSE → True
    assert policy_action_accuracy(["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"], Q_MULTI["expected_route"]) is True
    # multi-hop 但没拆解 → False
    assert policy_action_accuracy(["RETRIEVE", "ACCEPT"], Q_MULTI["expected_route"]) is False
    # abstain 类
    assert policy_action_accuracy(["RETRIEVE", "ABSTAIN"], ["RETRIEVE", "ABSTAIN"]) is True


def test_retry_recovery():
    assert retry_recovery(["RETRIEVE", "RETRIEVE", "ACCEPT"], True) is True
    assert retry_recovery(["RETRIEVE", "ACCEPT"], True) is False  # 无循环内 RETRIEVE
    assert retry_recovery(["RETRIEVE", "RETRIEVE", "ACCEPT"], False) is False  # 没命中


def test_decomposition_success():
    assert decomposition_success(["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"], True) is True
    assert decomposition_success(["RETRIEVE", "DECOMPOSE", "ABSTAIN"], True) is False
    assert decomposition_success(["RETRIEVE", "ACCEPT"], True) is False


def test_unnecessary_action_rate():
    # easy 题过度思考 → True
    assert unnecessary_action_rate(["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"], "easy_single_hop") is True
    # easy 直接答 → False
    assert unnecessary_action_rate(["RETRIEVE", "ACCEPT"], "easy_single_hop") is False
    # 非 easy 类不算
    assert unnecessary_action_rate(["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"], "multi_hop_composition") is False


def test_compute_capability_metrics_smoke():
    """汇总函数 smoke test：rescue / ood / unnecessary 判定正确"""
    cases = [
        {
            "question": {
                "id": "t1",
                "type": "easy_single_hop",
                "hops": [{"gold_chunk_ids": ["a"]}],
                "final_answer": "888",
                "expected_route": ["RETRIEVE", "ACCEPT"],
            },
            "v0_sources": [{"id": "a"}],
            "v1_sources": [{"id": "a"}],
            "v1_route": ["RETRIEVE", "ACCEPT"],
            "v1_answer": "888",
            "v1_abstained": False,
        },
        # V0 miss → Agent hit = Rescue
        {
            "question": {
                "id": "t2",
                "type": "multi_hop_composition",
                "hops": [{"gold_chunk_ids": ["a"]}, {"gold_chunk_ids": ["b"]}],
                "final_answer": "X",
                "expected_route": ["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"],
            },
            "v0_sources": [{"id": "z"}],
            "v1_sources": [{"id": "a"}, {"id": "b"}],
            "v1_route": ["RETRIEVE", "DECOMPOSE", "RETRIEVE", "ACCEPT"],
            "v1_answer": "X",
            "v1_abstained": False,
        },
        {
            "question": {
                "id": "t3",
                "type": "unsupported_ood",
                "hops": [],
                "final_answer": "",
                "expected_route": ["RETRIEVE", "ABSTAIN"],
            },
            "v0_sources": [],
            "v1_sources": [],
            "v1_route": ["RETRIEVE", "ABSTAIN"],
            "v1_answer": "",
            "v1_abstained": True,
        },
    ]
    m = compute_agent_capability_metrics(cases)
    assert m["final_rescue"] == 1
    assert m["harm"] == 0
    assert m["net_utility"] == 1
    assert m["ood_reject"] == "1/1"
    assert m["false_abstain"] == "0/2"
    assert m["decomposition_success"] == 1
    assert m["final_answer_accuracy"] == "2/3"

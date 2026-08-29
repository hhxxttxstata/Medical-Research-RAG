"""
Agentic RAG v1 单元测试

覆盖核心决策逻辑（不依赖真实 LLM/检索）：
  - _decide: grader 输出 → 决策映射
  - _dedup_accumulate: 候选去重累积
  - classify 相关: route 生成
  - _build_retrieval_variant: 检索变体构造
  - state_to_dict: 序列化

注：LLM grader / decompose 依赖外部服务，不在此测（集成测试见 scripts/step10_agentic_eval.py）
"""

import pytest

from src.agentic_rag import AgenticRAG, AgentState


@pytest.fixture
def agent():
    return AgenticRAG(retriever=None, generator=None)


def test_decide_mapping(agent):
    """grader 输出 → (evidence_status, action) 映射"""
    assert agent._decide({"decision": "sufficient"}) == ("SUFFICIENT", "ACCEPT")
    assert agent._decide({"decision": "insufficient"}) == ("INSUFFICIENT", "RETRIEVE")
    assert agent._decide({"decision": "needs_decomposition"}) == ("INSUFFICIENT", "DECOMPOSE")
    assert agent._decide({"decision": "insufficient", "unsupported": True}) == ("UNSUPPORTED", "ABSTAIN")


def test_dedup_accumulate(agent):
    """候选按 id 去重累积"""
    state = AgentState()
    agent._dedup_accumulate(state, [{"id": "a"}, {"id": "b"}])
    agent._dedup_accumulate(state, [{"id": "b"}, {"id": "c"}])
    assert [c["id"] for c in state.candidates] == ["a", "b", "c"]


def test_build_retrieval_variant(agent):
    """检索变体：从候选提取高频术语"""
    state = AgentState()
    state.candidates = [
        {"id": "1", "text": "肺栓塞 CTPA 诊断 肺栓塞 治疗 肺栓塞 抗凝"},
        {"id": "2", "text": "肺栓塞 溶栓 方案 抗凝 治疗"},
    ]
    variant = agent._build_retrieval_variant(state, "肺栓塞如何治疗？")
    assert "肺栓塞如何治疗？" in variant  # 原问题保留


def test_state_to_dict_serializable(agent):
    """State 可 JSON 序列化"""
    state = AgentState(original_query="测试问题", evidence_score=0.8, route=["RETRIEVE", "ACCEPT"])
    state.candidates = [{"id": "a", "metadata": {"filename": "x.md"}, "text": "内容"}]
    d = agent.state_to_dict(state)
    import json

    json.dumps(d, ensure_ascii=False)  # 不抛异常即可
    assert d["original_query"] == "测试问题"
    assert d["route"] == ["RETRIEVE", "ACCEPT"]


def test_is_topically_related_med(agent):
    """医学问题：候选含区分性实体 → 相关"""
    from src.agentic_rag import _GENERIC_TERMS  # noqa: F401

    candidates = [{"id": "1", "text": "本方案针对胸部 CT 影像中的肺结节检测任务"}]
    assert agent._is_topically_related("CT影像肺结节的检测流程是怎样的？", candidates) is True


def test_is_topically_related_ood(agent):
    """OOD 问题：候选不含区分性实体 → 不相关（拒答）"""
    candidates = [{"id": "1", "text": "肺栓塞是一种常见的致命性疾病，其诊断依赖于CTPA检查。"}]
    # "糖尿病肾病" 的实体在候选里没有
    assert agent._is_topically_related("糖尿病肾病患者的血压控制目标是多少？", candidates) is False
    # 明确 OOD 关键词 → 直接 False
    assert agent._is_topically_related("如何配置Kubernetes集群的RBAC权限？", candidates) is False


def test_is_multi_part(agent):
    """结构判断（B1 契约）：_is_multi_part 只保留多问号可靠信号"""
    # 多问号 → 多独立子问题
    assert agent._is_multi_part("急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？") is True
    assert agent._is_multi_part("肺栓塞的诊断标准是什么？溶栓治疗的适应症有哪些？") is True
    # 同主题追问（共享实体）→ 不算
    assert agent._is_multi_part("DICOM是什么？转换公式是什么？") is False
    # 极短裸问（≤8 字符无新主题名词）→ 追问而非独立子问题
    assert agent._is_multi_part("敏感度要求是多少？推理框架是什么？") is False
    # 单问号简单题 → 不算
    assert agent._is_multi_part("sPESI评分中收缩压低于多少mmHg记1分？") is False
    assert agent._is_multi_part("肺栓塞如何治疗？") is False
    # B1：对比词/并列词不再计入（移入 _is_comparison）
    assert agent._is_multi_part("U-Net和TransUNet在医学图像分割中各有什么优势和局限？") is False


def test_is_comparison(agent):
    """对比/并列结构信号（B1）：走 LLM grader 裁决，不直接 DECOMPOSE"""
    # 对比词（可能单跳可答，如"窗宽和窗位的区别"）
    assert agent._is_comparison("U-Net和TransUNet在医学图像分割中各有什么优势和局限？") is True
    assert agent._is_comparison("急性肺栓塞和慢性肺栓塞有什么区别？") is True
    assert agent._is_comparison("窗宽和窗位的区别是什么？") is True
    # 并列 + 英文专名 → 不同实体
    assert agent._is_comparison("CTPA与MRPA在肺栓塞诊断中分别有什么优势？") is True
    # 非对比 → False
    assert agent._is_comparison("sPESI评分中收缩压低于多少mmHg记1分？") is False
    assert agent._is_comparison("肺栓塞如何治疗？") is False
    assert agent._is_comparison("敏感度要求是多少？推理框架是什么？") is False


def test_entity_overlap(agent):
    """实体覆盖率信号：相关问题高，无关问题低"""
    cands = [{"id": "1", "text": "CT影像肺结节检测使用LUNA16数据集，预处理包括窗宽窗位调整、体素归一化。"}]
    high = agent._entity_overlap("CT影像肺结节的检测流程是怎样的？", cands)
    low = agent._entity_overlap("如何配置Kubernetes集群的RBAC权限？", cands)
    assert high > 0.0
    assert low < 0.3


# ── Policy Node counterfactual（stub reranker，不依赖 LLM）──


class _StubReranker:
    """确定性 reranker stub：按文本是否含"答案"关键词给分"""

    model_ready = True

    def __init__(self, top1_score: float):
        self._top1 = top1_score

    def rerank(self, question, candidates, k):
        out = []
        for i, c in enumerate(candidates[:k]):
            out.append({"id": c["id"], "text": c["text"], "_rerank_score": self._top1 if i == 0 else self._top1 * 0.9})
        return out


def _make_agent(reranker) -> "AgenticRAG":
    from src.agentic_rag import AgenticRAG

    class _NoLLM:
        """LLM 不可用 stub：chat 抛异常 → policy 走 rule fallback"""

        def chat(self, *a, **kw):
            raise RuntimeError("no llm")

    return AgenticRAG(retriever=None, generator=_NoLLM(), reranker=reranker, max_iterations=2)


def test_policy_signal_accept():
    """top1 高相关（≥0.5）→ signal ACCEPT（即使 grader 说 insufficient）"""
    agent = _make_agent(_StubReranker(top1_score=0.9))
    from src.agentic_rag import AgentState

    state = AgentState(original_query="CT影像肺结节的检测流程是怎样的？")
    state.candidates = [{"id": "a", "text": "CT影像肺结节检测预处理流程包括窗宽窗位调整、体素归一化。"}]
    state.iteration = 1
    status, action, mode = agent.policy(
        "CT影像肺结节的检测流程是怎样的？", state, {"decision": "insufficient", "evidence_score": 0.3}
    )
    assert action == "ACCEPT" and mode == "signal" and status == "SUFFICIENT"


def test_policy_signal_abstain():
    """top1 极低（<0.05）+ 迭代用尽 → signal ABSTAIN"""
    agent = _make_agent(_StubReranker(top1_score=0.01))
    from src.agentic_rag import AgentState

    state = AgentState(original_query="如何配置Kubernetes集群的RBAC权限？")
    state.candidates = [{"id": "a", "text": "肺栓塞诊断依赖CTPA检查。"}]
    state.iteration = 2  # 迭代用尽
    status, action, mode = agent.policy(
        "如何配置Kubernetes集群的RBAC权限？", state, {"decision": "insufficient", "evidence_score": 0.1}
    )
    assert action == "ABSTAIN" and mode == "signal" and status == "UNSUPPORTED"


def test_policy_signal_retrieve_when_iteration_left():
    """top1 极低 + 迭代未用尽 → signal RETRIEVE（给二次检索机会）"""
    agent = _make_agent(_StubReranker(top1_score=0.01))
    from src.agentic_rag import AgentState

    state = AgentState(original_query="如何配置Kubernetes集群的RBAC权限？")
    state.candidates = [{"id": "a", "text": "肺栓塞诊断依赖CTPA检查。"}]
    state.iteration = 1
    status, action, mode = agent.policy(
        "如何配置Kubernetes集群的RBAC权限？", state, {"decision": "insufficient", "evidence_score": 0.1}
    )
    assert action == "RETRIEVE" and mode == "signal"


def test_policy_signal_multi_part_decompose():
    """v2: multi-part 问题 → 直接 DECOMPOSE（不再交 LLM 自由选择）"""
    agent = _make_agent(_StubReranker(top1_score=0.9))
    from src.agentic_rag import AgentState

    state = AgentState(original_query="急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？")
    state.candidates = [{"id": "a", "text": "急性肺栓塞核心病理是血栓阻塞肺动脉；慢性肺栓塞机化转化为CTEPH。"}]
    state.iteration = 1
    status, action, mode = agent.policy(
        "急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？",
        state,
        {"decision": "insufficient", "evidence_score": 0.3},
    )
    assert action == "DECOMPOSE" and mode == "signal"


# ── Step 13 v2: Hop State / Accumulator / Completeness ──


class _HighReranker:
    """给特定 hop 高分的 reranker stub"""

    model_ready = True

    def __init__(self, hop_scores: dict[int, float]):
        self._scores = hop_scores

    def rerank(self, query, candidates, k):
        # 按 hop_id 给分（测试用 query 里带 hop 标记）；无标记 → 取最高分
        import re

        m = re.search(r"hop_(\d+)", query)
        score = self._scores.get(int(m.group(1)), 0.1) if m else (max(self._scores.values()) if self._scores else 0.9)
        out = []
        for _i, c in enumerate(candidates[:k]):
            out.append({"id": c["id"], "text": c.get("text", ""), "_rerank_score": score})
        return out


def _make_v2_agent(reranker) -> "AgenticRAG":
    from src.agentic_rag import AgenticRAG

    class _NoLLM:
        def chat(self, *a, **kw):
            raise RuntimeError("no llm")

    return AgenticRAG(retriever=None, generator=_NoLLM(), reranker=reranker, max_iterations=2)


def test_v2_accumulate_dedup():
    """Evidence Accumulator：跨轮去重累积"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9}))
    state = AgentState(original_query="q")
    agent._accumulate_evidence(state, [{"id": "a"}, {"id": "b"}])
    agent._accumulate_evidence(state, [{"id": "b"}, {"id": "c"}])
    assert [c["id"] for c in state.evidence_bank] == ["a", "b", "c"]
    assert [c["id"] for c in state.candidates] == ["a", "b", "c"]


def test_v2_hops_init_and_assign():
    """plan → HopState + evidence_by_hop 分配"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.8}))
    state = AgentState(original_query="q")
    plan = [
        {"hop_id": 1, "question": "hop_1 子问题", "depends_on": None, "status": "PENDING"},
        {"hop_id": 2, "question": "hop_2 子问题", "depends_on": None, "status": "PENDING"},
    ]
    agent._init_hops_from_plan(state, plan)
    assert len(state.hops) == 2
    assert state.hops[0].subquery == "hop_1 子问题"
    assert state.hops[0].support_status == "PENDING"
    # 分配证据
    agent._accumulate_evidence(state, [{"id": "c1"}, {"id": "c2"}])
    assert state.evidence_by_hop[1]  # hop1 有证据
    assert state.evidence_by_hop[2]  # hop2 有证据


def test_v2_completeness_and_missing():
    """Completeness Check：SUPPORTED hop 计数 + MISSING hop 定位"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.02}))
    state = AgentState(original_query="q")
    plan = [
        {"hop_id": 1, "question": "hop_1 子问题", "depends_on": None, "status": "PENDING"},
        {"hop_id": 2, "question": "hop_2 子问题", "depends_on": None, "status": "PENDING"},
    ]
    agent._init_hops_from_plan(state, plan)
    agent._accumulate_evidence(state, [{"id": "c1"}, {"id": "c2"}])
    completeness, missing = agent._compute_completeness(state)
    assert completeness == 0.5  # hop1 supported, hop2 missing
    assert len(missing) == 1
    assert missing[0].hop_id == 2
    assert state.hops[0].support_status == "SUPPORTED"
    assert state.hops[1].support_status == "MISSING"


def test_v2_find_missing_hop():
    """定位第一个 MISSING hop（targeted retrieve 的目标）"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.02}))
    state = AgentState(original_query="q")
    agent._init_hops_from_plan(
        state,
        [
            {"hop_id": 1, "question": "hop_1", "depends_on": None, "status": "PENDING"},
            {"hop_id": 2, "question": "hop_2", "depends_on": None, "status": "PENDING"},
        ],
    )
    agent._accumulate_evidence(state, [{"id": "c1"}])
    hop = agent._find_missing_hop(state)
    assert hop is not None and hop.hop_id == 2


def test_v2_policy_unsupported_overrides_high_relevance():
    """13C 核心：grader UNSUPPORTED 时即使 top1 高相关也不 ACCEPT（bh_ood_02 修复）"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_StubReranker(top1_score=0.946))
    state = AgentState(original_query="2026年ESC年会发布了哪些新指南？")
    state.candidates = [{"id": "a", "text": "诊断流程指南推荐YEARS模型，2025年指南。"}]
    state.iteration = 1
    state.retrieval_budget = 2
    # grader 判 insufficient + reason 提示"未提及" → 不应 ACCEPT
    status, action, mode = agent.policy(
        "2026年ESC年会发布了哪些新指南？",
        state,
        {"decision": "insufficient", "evidence_score": 0.1, "reason": "证据未提及2026年ESC年会"},
    )
    assert action != "ACCEPT"
    assert mode == "signal"


def test_v2_policy_completeness_accept():
    """plan 全部 SUPPORTED + top1 高相关 → ACCEPT"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.85}))
    state = AgentState(original_query="q")
    agent._init_hops_from_plan(
        state,
        [
            {"hop_id": 1, "question": "hop_1", "depends_on": None, "status": "PENDING"},
            {"hop_id": 2, "question": "hop_2", "depends_on": None, "status": "PENDING"},
        ],
    )
    state.candidates = [{"id": "c1"}, {"id": "c2"}]
    state.iteration = 2
    state.retrieval_budget = 2
    agent._accumulate_evidence(state, [{"id": "c1"}, {"id": "c2"}])
    status, action, mode = agent.policy("q", state, {"decision": "insufficient", "evidence_score": 0.3})
    assert action == "ACCEPT" and status == "SUFFICIENT"


def test_v2_policy_completeness_priority_over_unsupported():
    """plan 全 SUPPORTED 时，grader 说'缺失'也不 ABSTAIN（bh_multi_01 修复）"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.85}))
    state = AgentState(original_query="敏感度要求是多少？推理框架是什么？")
    agent._init_hops_from_plan(
        state,
        [
            {"hop_id": 1, "question": "hop_1", "depends_on": None, "status": "PENDING"},
            {"hop_id": 2, "question": "hop_2", "depends_on": None, "status": "PENDING"},
        ],
    )
    state.iteration = 3
    state.retrieval_budget = 1
    agent._accumulate_evidence(state, [{"id": "c1"}, {"id": "c2"}])
    # grader 说 insufficient + reason 含"未提及"（模拟 bh_multi_01）
    status, action, mode = agent.policy(
        "敏感度要求是多少？推理框架是什么？",
        state,
        {"decision": "insufficient", "evidence_score": 0.3, "reason": "证据未提及推理加速框架"},
    )
    assert action == "ACCEPT"  # completeness 优先于 unsupported


def test_v2_policy_multi_part_trigger_decompose():
    """multi-part 且无 plan → DECOMPOSE（13E）"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_StubReranker(top1_score=0.9))
    state = AgentState(original_query="U-Net 和 TransUNet 的区别是什么？它们的应用场景分别是什么？")
    state.candidates = [{"id": "a", "text": "内容"}]
    state.iteration = 1
    state.retrieval_budget = 3
    status, action, mode = agent.policy(
        "U-Net 和 TransUNet 的区别是什么？它们的应用场景分别是什么？",
        state,
        {"decision": "insufficient", "evidence_score": 0.3},
    )
    assert action == "DECOMPOSE" and mode == "signal"


def test_v2_final_evidence_merges_hops():
    """有 plan 时 final_evidence 从 evidence_by_hop 合并（bh_partial_01 修复）"""
    from src.agentic_rag import AgentState

    agent = _make_v2_agent(_HighReranker({1: 0.9, 2: 0.85}))
    state = AgentState(original_query="q")
    agent._init_hops_from_plan(
        state,
        [
            {"hop_id": 1, "question": "hop_1", "depends_on": None, "status": "PENDING"},
            {"hop_id": 2, "question": "hop_2", "depends_on": None, "status": "PENDING"},
        ],
    )
    agent._accumulate_evidence(state, [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}])
    ev = agent._select_final_evidence("q", state.candidates, 5, state=state)
    # hop1 分到 c1,c2,c3 中 top3，hop2 同理；合并去重后应含所有证据
    assert len(ev) >= 1
    assert all(c["id"] in {x["id"] for x in ev} for c in ev)

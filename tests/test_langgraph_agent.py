"""
Step 16: LangGraph Runtime Adapter 单元测试

覆盖（不依赖真实 LLM/检索）：
  - 图结构：节点完整、条件边路由正确
  - _route：预算耗尽强制 ABSTAIN（与自定义 runner 的 while 条件一致）
  - 节点行为：retrieve 累积 / decompose 拆解失败退化 / finalize 终局
  - 与自定义 runner 的 parity（stub retriever + stub reranker，确定性）

注：真实 index 上的 parity 实验见 scripts/step16_runtime_parity.py
"""

from src.agentic_rag import AgentState
from src.langgraph_agent import GraphState, LangGraphAgenticRAG

# ── Stub 组件（确定性，无 LLM / 无 index）──


class _StubRetriever:
    """确定性检索：返回 3 个固定 chunk（含 gold 标记）"""

    def __init__(self):
        self.chunks = [
            {"id": "c1", "text": "sPESI评分中收缩压<100mmHg记1分。年龄>80岁记1分。灵敏度96%。", "score": 0.8},
            {"id": "c2", "text": "肺栓塞抗凝治疗使用低分子肝素。", "score": 0.6},
            {"id": "c3", "text": "急性肺栓塞CT表现：马鞍征、环征。", "score": 0.5},
        ]

    def _hybrid_retrieve(self, query, fetch_k=20):
        return list(self.chunks)


class _StubReranker:
    """确定性 reranker：按文本是否含 query 关键实体给分"""

    model_ready = True

    def rerank(self, query, candidates, k):
        import re

        m = re.search(r"hop_(\d+)", query)
        score = 0.9 if m else 0.7
        out = []
        for i, c in enumerate(candidates[:k]):
            out.append({"id": c["id"], "text": c.get("text", ""), "_rerank_score": score if i == 0 else score * 0.8})
        return out


class _NoLLM:
    """LLM 不可用 stub：chat 抛异常 → policy/grader 走规则 fallback"""

    def chat(self, *a, **kw):
        raise RuntimeError("no llm")

    def generate_structured(self, *a, **kw):
        raise RuntimeError("no llm")


def _make_agent():
    from src.agentic_rag import AgenticRAG

    return AgenticRAG(retriever=_StubRetriever(), generator=_NoLLM(), reranker=_StubReranker(), max_iterations=2)


def _make_cost_aware_agent():
    """v2.1 cost-aware agent（stub 组件，无 LLM / 无 index）"""
    from src.cost_aware_agentic_rag import CostAwareAgenticRAG

    return CostAwareAgenticRAG(
        retriever=_StubRetriever(), generator=_NoLLM(), reranker=_StubReranker(), max_iterations=2
    )


# ── 图结构 ──


def test_graph_nodes_registered():
    """StateGraph 应包含 5 个节点 + START/END 边"""
    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    nodes = set(lg._graph.get_graph().nodes.keys())
    assert {"retrieve", "decompose", "evaluate", "policy", "finalize"} <= nodes


def test_route_budget_exhausted():
    """预算耗尽时 RETRIEVE/DECOMPOSE → 强制 ABSTAIN（与自定义 while 条件一致）"""
    st = AgentState(original_query="q")
    st.retrieval_budget = 0
    gs: GraphState = {"state": st, "decision": "RETRIEVE", "action_mode": "", "reason": "", "final": None}
    assert LangGraphAgenticRAG._route(gs) == "ABSTAIN"
    gs["decision"] = "ACCEPT"
    assert LangGraphAgenticRAG._route(gs) == "ACCEPT"


# ── 节点行为 ──


def test_retrieve_node_initial():
    """初始 retrieve：累积证据 + 迭代计数 + 预算扣减"""
    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    st = AgentState(original_query="sPESI评分中收缩压低于多少mmHg记1分？")
    st.retrieval_budget = 4
    gs: GraphState = {"state": st, "decision": "", "action_mode": "", "reason": "", "final": None, "_target_hop": None}
    out = lg._retrieve_node(gs)
    assert len(st.evidence_bank) == 3
    assert st.iteration == 1
    assert st.retrieval_budget == 3


def test_decompose_node_failure_fallback():
    """拆解失败（LLM 不可用）→ 退化为单跳再检索（与自定义 runner 一致）"""
    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    st = AgentState(original_query="U-Net 和 TransUNet 的区别是什么？")
    st.retrieval_budget = 3
    st.route.append("RETRIEVE")
    gs: GraphState = {"state": st, "decision": "DECOMPOSE", "action_mode": "", "reason": "", "final": None}
    out = lg._decompose_node(gs)
    assert st.hops == []  # 拆解失败清空 plan
    assert st.decompose_attempted is True
    assert len(st.retrieval_history) == 1  # retry 检索
    assert st.retrieval_budget == 2


def test_finalize_abstain():
    """finalize：ABSTAIN → 拒答回答 + abstained=True"""
    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    st = AgentState(original_query="如何配置Kubernetes集群？")
    st.candidates = [{"id": "c1", "text": "肺栓塞诊断依赖CTPA。"}]
    gs: GraphState = {"state": st, "decision": "ABSTAIN", "action_mode": "signal", "reason": "证据不足", "final": None}
    out = lg._finalize_node(gs)
    assert out["final"]["abstained"] is True
    assert "知识库中未找到" in out["final"]["answer"]


# ── Parity：Custom vs LangGraph（确定性 stub 环境）──


def test_parity_single_hop_accept():
    """单跳易题：两种 runtime 应走相同 route 并给出相同终局"""

    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    q = "sPESI评分中收缩压低于多少mmHg记1分？"
    r1 = agent.run(q, fetch_k=5, verbose=False)
    r2 = lg.run(q, fetch_k=5, verbose=False)
    assert r1["route"] == r2["route"], f"{r1['route']} != {r2['route']}"
    assert r1["abstained"] == r2["abstained"]
    assert r1["iterations"] == r2["iterations"]
    # stub 环境无 LLM 生成：若 ACCEPT 则生成失败标记，双方应一致
    assert r1["answer"] == r2["answer"] or (r1["abstained"] and r2["abstained"])


def test_parity_multi_hop_route():
    """multi-part 题：两种 runtime 的 route 一致（DECOMPOSE 或同等决策序列）"""

    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    q = "急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？"
    r1 = agent.run(q, fetch_k=5, verbose=False)
    r2 = lg.run(q, fetch_k=5, verbose=False)
    assert r1["route"] == r2["route"], f"{r1['route']} != {r2['route']}"
    assert r1["abstained"] == r2["abstained"]


def test_parity_ood_abstain():
    """OOD 题：两种 runtime 的 route 完全一致（本 stub 无 LLM，行为以 route 对齐为准）"""
    agent = _make_agent()
    lg = LangGraphAgenticRAG(agent)
    q = "如何用 Python 的 asyncio 实现高性能 WebSocket 服务器？"
    r1 = agent.run(q, fetch_k=5, verbose=False)
    r2 = lg.run(q, fetch_k=5, verbose=False)
    assert r1["route"] == r2["route"]
    assert r1["abstained"] == r2["abstained"]
    assert r1["answer"] == r2["answer"]


# ── B2 / B4：cost-aware precheck + grader_called parity ──


def test_graph_precheck_node_cost_aware():
    """B2：cost-aware 图含 precheck 节点，且无 evaluate 节点（门控内置）"""
    agent = _make_cost_aware_agent()
    lg = LangGraphAgenticRAG(agent)
    nodes = set(lg._graph.get_graph().nodes.keys())
    assert {"precheck", "retrieve", "decompose", "policy", "finalize"} <= nodes
    assert "evaluate" not in nodes


def test_precheck_node_routes():
    """B2：precheck 节点——多问号 → DECOMPOSE 直进（未检索）；普通 → 初始检索"""
    agent = _make_cost_aware_agent()
    lg = LangGraphAgenticRAG(agent)
    st = AgentState(original_query="肺栓塞的诊断标准是什么？溶栓治疗的适应症有哪些？")
    st.retrieval_budget = 4
    gs: GraphState = {"state": st, "decision": "", "action_mode": "", "reason": "", "final": None, "_target_hop": None}
    out = lg._precheck_node(gs)
    assert out["decision"] == "DECOMPOSE"
    assert st.route == ["DECOMPOSE"]
    assert st.retrieval_history == []  # 预检不触发检索

    st2 = AgentState(original_query="sPESI评分中收缩压低于多少mmHg记1分？")
    st2.retrieval_budget = 4
    gs2: GraphState = {
        "state": st2,
        "decision": "",
        "action_mode": "",
        "reason": "",
        "final": None,
        "_target_hop": None,
    }
    out2 = lg._precheck_node(gs2)
    assert out2["decision"] == ""
    assert st2.route == ["RETRIEVE"]


def test_policy_node_grader_called_accumulates():
    """B4：多轮决策中 grader_called 跨轮累计（第二轮 cheap 决策不覆盖第一轮）"""
    agent = _make_cost_aware_agent()
    lg = LangGraphAgenticRAG(agent)
    st = AgentState(original_query="肺栓塞如何治疗？")
    st.retrieval_budget = 4
    gs: GraphState = {
        "state": st,
        "decision": "",
        "action_mode": "",
        "reason": "",
        "final": None,
        "_grade": {"decision": "insufficient", "reason": "initial", "evidence_score": 0.0},
        "_target_hop": None,
        "_grader_called": False,
    }
    # 第一轮：证据词面零重叠 → UNCERTAIN → LLM grader 被调（stub 走规则 fallback）
    st.evidence_bank = [{"id": "x", "text": "unrelated totally different english text", "score": 0.8}]
    st.candidates = list(st.evidence_bank)
    lg._policy_node(gs)
    assert gs["_grader_called"] is True
    # 第二轮：词面重叠 + top1 高 → cheap ACCEPT（不调 grader）→ 标志必须保持 True
    st.evidence_bank = [{"id": "y", "text": "肺栓塞抗凝治疗使用低分子肝素，肺栓塞 抗凝 方案。", "score": 0.8}]
    st.candidates = list(st.evidence_bank)
    gs["_grade"] = {"decision": "insufficient", "reason": "initial", "evidence_score": 0.0}
    lg._policy_node(gs)
    assert gs["_grader_called"] is True


def test_parity_cost_aware_precheck_decompose():
    """B2：cost-aware 多问号题 → 预检直进 DECOMPOSE，两种 runtime route 一致"""
    agent = _make_cost_aware_agent()
    lg = LangGraphAgenticRAG(agent)
    q = "急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？"
    r1 = agent.run(q, fetch_k=5, verbose=False)
    r2 = lg.run(q, fetch_k=5, verbose=False)
    assert r1["route"] == r2["route"], f"{r1['route']} != {r2['route']}"
    assert r1["abstained"] == r2["abstained"]
    assert r1["iterations"] == r2["iterations"]
    assert r1["route"][0] == "DECOMPOSE"  # B2：跳过初始 RETRIEVE
    # B4：grader 标志两 runtime 一致（本路径 cheap DECOMPOSE → 无 grader）
    assert r1["observation"]["grader_called"] == r2["grader_called"]


def test_parity_cost_aware_single_hop():
    """cost-aware 单跳题：两种 runtime route / iterations / grader 标志一致"""
    agent = _make_cost_aware_agent()
    lg = LangGraphAgenticRAG(agent)
    q = "sPESI评分中收缩压低于多少mmHg记1分？"
    r1 = agent.run(q, fetch_k=5, verbose=False)
    r2 = lg.run(q, fetch_k=5, verbose=False)
    assert r1["route"] == r2["route"], f"{r1['route']} != {r2['route']}"
    assert r1["abstained"] == r2["abstained"]
    assert r1["iterations"] == r2["iterations"]
    assert r1["observation"]["grader_called"] == r2["grader_called"]

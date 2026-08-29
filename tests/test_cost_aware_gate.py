"""
B1/B2/B3 修复的 cheap gate 单元测试（stub 环境，无 LLM / 无索引）

覆盖：
  - B1：对比/并列 → UNCERTAIN（LLM grader 裁决），多问号 → DECOMPOSE，
        已拆解过不再重复触发
  - B3：OOD 关键词 + top1 极低 → 直接 ABSTAIN（不浪费一轮 RETRIEVE）
  - B2：run() 结构预检——多问号跳过全问题初始检索（route 首元素 DECOMPOSE）
"""

from src.agentic_rag import AgentState
from src.cost_aware_agentic_rag import CostAwareAgenticRAG


class _StubRetriever:
    def _hybrid_retrieve(self, query, fetch_k=20):
        return [
            {"id": "c1", "text": "sPESI评分中收缩压<100mmHg记1分。年龄>80岁记1分。灵敏度96%。", "score": 0.8},
            {"id": "c2", "text": "肺栓塞抗凝治疗使用低分子肝素。", "score": 0.6},
            {"id": "c3", "text": "急性肺栓塞CT表现：马鞍征、环征。", "score": 0.5},
        ]


class _RecordingRetriever(_StubRetriever):
    """记录检索 query，验证 B2 预检跳过初始检索"""

    def __init__(self):
        self.queries: list[str] = []

    def _hybrid_retrieve(self, query, fetch_k=20):
        self.queries.append(query)
        return super()._hybrid_retrieve(query, fetch_k)


class _StubReranker:
    model_ready = True

    def __init__(self, top1: float = 0.9):
        self._top1 = top1

    def rerank(self, query, candidates, k):
        out = []
        for i, c in enumerate(candidates[:k]):
            out.append(
                {"id": c["id"], "text": c.get("text", ""), "_rerank_score": self._top1 if i == 0 else self._top1 * 0.7}
            )
        return out


class _NoLLM:
    def chat(self, *a, **kw):
        raise RuntimeError("no llm")

    def generate_structured(self, *a, **kw):
        raise RuntimeError("no llm")


def _gate(question, bank=None, hops=None, budget=4, iteration=0, top1=0.9, decompose_attempted=False):
    """构造 state 并返回 _signal_verdict 的 (verdict, reason)"""
    agent = CostAwareAgenticRAG(
        retriever=_StubRetriever(), generator=_NoLLM(), reranker=_StubReranker(top1=top1), max_iterations=2
    )
    state = AgentState(original_query=question)
    state.evidence_bank = bank or []
    state.candidates = list(state.evidence_bank)
    state.hops = hops or []
    state.retrieval_budget = budget
    state.iteration = iteration
    state.decompose_attempted = decompose_attempted
    return agent._signal_verdict(state, question)


# ── B1：结构信号两级化 ──


def test_gate_multi_question_decompose():
    """多问号 → 直接 DECOMPOSE（可靠结构信号，不看证据）"""
    verdict, _ = _gate("肺栓塞的诊断标准是什么？溶栓治疗的适应症有哪些？")
    assert verdict == "DECOMPOSE"


def test_gate_comparison_uncertain():
    """B1：对比/并列 → UNCERTAIN（交 grader 裁决），即使 top1 高相关也不 cheap ACCEPT"""
    verdict, reason = _gate("急性肺栓塞和慢性肺栓塞有什么区别？")
    assert verdict == "UNCERTAIN"
    assert "对比/并列" in reason


def test_gate_comparison_after_decompose_not_retriggered():
    """B1：已拆解过（decompose_attempted）→ 对比信号不再触发 UNCERTAIN，
    回到证据信号路径（词面重叠 + top1 高 → ACCEPT）"""
    verdict, _ = _gate(
        "急性肺栓塞和慢性肺栓塞有什么区别？",
        bank=[{"id": "c3", "text": "急性肺栓塞CT表现：马鞍征、环征。肺栓塞 慢性 区别"}],
        decompose_attempted=True,
    )
    assert verdict == "ACCEPT"


# ── B3：OOD 早拒 ──


def test_gate_ood_low_top1_abstain():
    """B3：OOD 关键词命中 + top1 极低 → 直接 ABSTAIN（不浪费一轮 RETRIEVE）"""
    verdict, reason = _gate("如何配置Kubernetes集群的RBAC权限？", top1=0.01, budget=3, iteration=1)
    assert verdict == "ABSTAIN"
    assert "领域外" in reason


def test_gate_non_ood_low_top1_retrieve():
    """对照：非 OOD + top1 极低 + 预算/迭代未耗尽 → 仍 RETRIEVE"""
    verdict, _ = _gate("肺栓塞如何治疗？", bank=[{"id": "x", "text": "unrelated"}], top1=0.01, budget=3, iteration=1)
    assert verdict == "RETRIEVE"


def test_gate_ood_high_top1_conflict_uncertain():
    """对照：OOD 关键词 + top1 高相关（e5 语义近邻）→ CONFLICT → UNCERTAIN（grader 裁决）"""
    verdict, _ = _gate("如何配置Kubernetes集群的RBAC权限？", bank=[{"id": "x", "text": "unrelated"}], top1=0.9)
    assert verdict == "UNCERTAIN"


# ── B2：run() 结构预检 ──


def test_run_precheck_skips_initial_retrieval():
    """B2：多问号题跳过全问题初始检索——stub LLM 拆解失败后仅 1 次退化检索，
    route 以 DECOMPOSE 开头（旧实现为 RETRIEVE→DECOMPOSE 两次检索）"""
    retriever = _RecordingRetriever()
    agent = CostAwareAgenticRAG(retriever=retriever, generator=_NoLLM(), reranker=_StubReranker(), max_iterations=2)
    result = agent.run("急性肺栓塞和慢性肺栓塞有什么区别？CTPA影像如何鉴别？", fetch_k=5, verbose=False)
    assert result["route"][0] == "DECOMPOSE"
    # 拆解失败 → 仅退化检索 1 次；旧实现是初始检索 + 退化检索 = 2 次
    assert len(retriever.queries) == 1
    assert result["observation"]["retrieval_calls"] == 1


def test_run_normal_initial_retrieval():
    """对照：非多问号 → 正常初始检索，route 以 RETRIEVE 开头"""
    retriever = _RecordingRetriever()
    agent = CostAwareAgenticRAG(retriever=retriever, generator=_NoLLM(), reranker=_StubReranker(), max_iterations=2)
    result = agent.run("sPESI评分中收缩压低于多少mmHg记1分？", fetch_k=5, verbose=False)
    assert result["route"][0] == "RETRIEVE"
    assert len(retriever.queries) == 1


# ── RERANK_CAP 截断修复（2026-08-17）──


class _KeywordReranker:
    """按文本是否含 hop 关键词给分的 stub reranker"""

    model_ready = True

    def rerank(self, query, candidates, k):
        out = []
        for c in candidates:
            score = 0.9 if "推理加速" in c.get("text", "") else 0.1
            out.append({"id": c["id"], "text": c.get("text", ""), "_rerank_score": score})
        out.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return out[:k]


def test_assign_to_hops_rrf_cap_keeps_late_evidence():
    """RERANK_CAP 截断必须按 RRF 分排序——hop 定向检索新捞的证据（bank append
    序末尾）不能被 [:16] 挤掉（bh_multi_01 实测：修复前 hop2 证据永远 MISSING）"""
    from src.agentic_rag import HopState

    agent = CostAwareAgenticRAG(
        retriever=_StubRetriever(), generator=_NoLLM(), reranker=_KeywordReranker(), max_iterations=2
    )
    state = AgentState(original_query="q")
    # 20 个干扰 chunk（RRF 低分、append 在前）+ gold（RRF 高分、append 在末尾）
    bank = [{"id": f"noise_{i}", "text": "无关内容", "score": 0.01, "_rrf_score": 0.01} for i in range(20)]
    bank.append({"id": "gold_1", "text": "模型使用 TensorRT 进行推理加速", "score": 0.9, "_rrf_score": 0.9})
    state.evidence_bank = bank
    state.hops = [HopState(hop_id=1, subquery="该模型推理加速使用什么框架？", required=True)]
    agent._assign_to_hops(state)
    assert "gold_1" in state.hops[0].evidence_ids, (
        f"gold 被 RERANK_CAP 截断排除: evidence_ids={state.hops[0].evidence_ids}"
    )
    assert state.hops[0].evidence_score >= 0.5

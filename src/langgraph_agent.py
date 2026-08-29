"""
Agentic RAG v2 — LangGraph Runtime Adapter（Step 16）

目的：不迁移 Agentic RAG Core，只把自研 while-loop orchestration 套成标准
LangGraph StateGraph runtime，证明核心 policy / evidence-state 架构
framework-agnostic（同一套检索、hop 追踪、信号门控在两种 runtime 下行为一致）。

设计原则（final_step.md Step 16）：
  - Hybrid Retriever / Reranker / HopState / Evidence Bank / Completeness /
    Policy / Cost-aware Gate / Grader / Generator 全部不动
  - 唯一改动：把 `while ... if action == ...` 的控制流换成 StateGraph 节点
  - 状态在 LangGraph 的 state dict 中只存**引用**（AgentState 对象本身不变），
    节点间的状态流转 = 读写同一 AgentState —— 行为与自定义 runner 完全一致

图结构：
    START → [precheck（v2.1 专用）] → retrieve → policy → [conditional edge]
                                          ├─ ACCEPT → finalize → END
                                          ├─ RETRIEVE → retrieve
                                          └─ DECOMPOSE → decompose → retrieve
    ABSTAIN（budget 耗尽）→ finalize（拒答）
    precheck：可靠多问号 → 直接 decompose（B2，跳过全问题初始检索）；否则 retrieve

用法:
    from src.langgraph_agent import LangGraphAgenticRAG
    agent = LangGraphAgenticRAG(agent_v2)        # 包装已有的 AgenticRAG 实例
    result = agent.run(question, fetch_k=20)     # 与 agent.run() 相同签名
"""

from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from .agentic_rag import AgenticRAG

# ── LangGraph State：state dict 只存 AgentState 引用 + 本轮决策结果 ──


class GraphState(TypedDict):
    """LangGraph 的共享状态（TypedDict 契约）

    - state: AgentState 对象引用（核心状态，节点间共享同一对象）
    - decision: 当前策略决策（ACCEPT / RETRIEVE / DECOMPOSE / ABSTAIN）
    - action_mode: 决策来源（cheap_signal / llm / rule_fallback，v2.1 兼容）
    - reason: 决策原因
    - final: run() 返回的结果 dict
    - _grade: 上一轮 evidence_grade 结果（跨节点传递）
    - _target_hop: RETRIEVE 的目标 hop（13D，跨节点传递）
    - _grader_called: 本轮是否调用了 LLM grader（v2.1 cost-aware 门控）
    """

    state: Any
    decision: str
    action_mode: str
    reason: str
    final: dict | None
    _grade: dict | None
    _target_hop: dict | None
    _grader_called: bool


class LangGraphAgenticRAG:
    """Agentic RAG v2 的 LangGraph runtime 适配器

    包装一个已有的 AgenticRAG 实例（冻结的 v2 或 v2.1 均可），把其工具与
    决策逻辑挂到 StateGraph 节点上。节点内部全部调用被包装对象的方法——
    零策略逻辑复制，保证两种 runtime 行为一致（parity 的前提）。
    """

    def __init__(self, agent: AgenticRAG):
        self._agent = agent
        self._fetch_k = 20  # 默认检索深度（run() 可覆盖）
        self._graph = self._build_graph()

    # ══════════════════════════════════════════════════
    #  节点（每个节点只做一件事，行为与被包装 agent 的方法一一对应）
    # ══════════════════════════════════════════════════

    def _precheck_node(self, gs: GraphState) -> GraphState:
        """B2：结构预检（v2.1 cost-aware 专用）——可靠多问号 → 直接 DECOMPOSE

        与自定义 runner 的 iteration-0 预检一致：多问号是问题内在属性，
        无需检索即可判定 → 跳过全问题初始检索，直接进 decompose 节点
        （hop 定向检索覆盖全部证据）。route 在此记录初始动作（与自定义
        runner 的 iteration-0 route.append 一一对应）。
        """
        st = gs["state"]
        from .agentic_rag import AgenticRAG

        if AgenticRAG._is_multi_part(st.original_query) and not st.decompose_attempted:
            st.route.append("DECOMPOSE")
            gs["decision"] = "DECOMPOSE"
            gs["reason"] = "问题含多个独立子问题，先生成 hop plan"
        else:
            st.route.append("RETRIEVE")
            gs["decision"] = ""
        return gs

    def _retrieve_node(self, gs: GraphState) -> GraphState:
        """RETRIEVE：混合检索 → 证据累积（对应 AgenticRAG.run 的初始/再检索）"""
        st = gs["state"]
        q = st.original_query
        if not st.retrieval_history:
            # 初始检索
            reason = "initial"
            query = q
        else:
            # 再检索：有 target_hop 走 hop 定向，否则换角度
            target = gs.get("_target_hop")
            if target and st.hops:
                query = target["query"]
                reason = "targeted-hop"
            else:
                query = self._agent._build_retrieval_variant(st, q)
                reason = "retrieve"
        results = self._agent.hybrid_search(query, fetch_k=self._fetch_k, note=reason)
        st.retrieval_history.append({"query": query, "sources": results, "iteration": st.iteration, "reason": reason})
        self._agent._accumulate_evidence(st, results)
        st.iteration += 1
        st.retrieval_budget -= 1
        gs["_target_hop"] = None
        return gs

    def _decompose_node(self, gs: GraphState) -> GraphState:
        """DECOMPOSE：结构化拆解 → 逐 hop 定向检索（对应 run 的 DECOMPOSE 分支）"""
        st = gs["state"]
        st.decompose_attempted = True
        plan = self._agent.decompose_plan(st.original_query)
        if plan:
            self._agent._init_hops_from_plan(st, plan)
            for hop in st.hops:
                if st.retrieval_budget <= 0:
                    break
                self._agent._targeted_retrieve(st, hop, fetch_k=self._fetch_k)
                st.iteration += 1
                st.retrieval_budget -= 1
            completeness, missing = self._agent._compute_completeness(st)
            st.completeness = completeness
            if missing and st.retrieval_budget > 0:
                for hop in missing:
                    if st.retrieval_budget <= 0:
                        break
                    self._agent._targeted_retrieve(st, hop, fetch_k=self._fetch_k)
                    st.iteration += 1
                    st.retrieval_budget -= 1
                completeness, missing = self._agent._compute_completeness(st)
                st.completeness = completeness
        else:
            # 拆解失败 → 退化单跳再检索（与自定义 runner 一致）
            st.hops = []
            st.plan = []
            more = self._agent.hybrid_search(
                st.original_query, fetch_k=self._fetch_k, note="retry-after-decompose-fail"
            )
            st.retrieval_history.append(
                {"query": st.original_query, "sources": more, "iteration": st.iteration, "reason": "retry"}
            )
            self._agent._accumulate_evidence(st, more)
            st.iteration += 1
            st.retrieval_budget -= 1
        return gs

    def _evaluate_node(self, gs: GraphState) -> GraphState:
        """evaluate：evidence_grade（LLM grader + 规则 fallback，与 v2 相同）"""
        st = gs["state"]
        grade = self._agent.evidence_grade(st.original_query, st.candidates)
        gs["_grade"] = grade
        st.evidence_score = grade["evidence_score"]
        return gs

    def _policy_node(self, gs: GraphState) -> GraphState:
        """policy：v2/v2.1 统一策略入口

        - v2（AgenticRAG）：LLM grader 常开 + policy 决策（evaluate 节点已
          产出 grade）
        - v2.1（CostAwareAgenticRAG）：Cheap Signal Gate → 仅 UNCERTAIN 时
          按需调 LLM grader/policy —— 同一图结构，策略实现不同
        """
        st = gs["state"]
        grade = gs["_grade"] or {}
        if hasattr(self._agent, "_decide_once"):
            # v2.1 cost-aware 决策（grader 由门控按需调用；grade 原地更新）
            from .cost_aware_agentic_rag import CostObservation

            obs = CostObservation()
            decision, mode = self._agent._decide_once(st.original_query, st, grade, obs, self._fetch_k, False)
            status = {
                "ACCEPT": "SUPPORTED",
                "ABSTAIN": "UNSUPPORTED",
                "RETRIEVE": "INSUFFICIENT",
                "DECOMPOSE": "INSUFFICIENT",
            }[decision]
            st.evidence_status = status
            # B4：跨轮累计——obs 是单轮快照，任何一轮调过 grader 即为 True
            # （与自定义 runner 的跨轮 obs 语义一致，修复指标失真）
            gs["_grader_called"] = bool(gs.get("_grader_called", False)) or obs.grader_called
        else:
            status, decision, mode = self._agent.policy(st.original_query, st, grade)
            st.evidence_status = status
            gs["_grader_called"] = bool(gs.get("_grader_called", False))
        gs["decision"] = decision
        gs["action_mode"] = mode
        gs["reason"] = grade.get("reason", "")
        # 循环内动作追加到 route（与自定义 runner 的 while-top 一致；
        # ACCEPT/ABSTAIN 由 finalize 追加，与自定义 runner 的终局一致）
        if decision in ("RETRIEVE", "DECOMPOSE"):
            st.route.append(decision)
        # RETRIEVE 带 target_hop（13D）
        gs["_target_hop"] = grade.get("target_hop")
        return gs

    def _finalize_node(self, gs: GraphState) -> GraphState:
        """finalize：终局 ACCEPT → 生成；ABSTAIN → 拒答（与 run 的终局逻辑一致）"""
        st = gs["state"]
        decision = gs["decision"]
        if decision == "ACCEPT":
            st.decision = "ACCEPT"
            st.route.append("ACCEPT")
            st.final_evidence = self._agent._select_final_evidence(
                st.original_query, st.candidates, self._fetch_k, state=st
            )
            answer = self._agent.generate(st.original_query, st.final_evidence)
        else:
            st.decision = "ABSTAIN"
            st.route.append("ABSTAIN")
            st.abstain_reason = gs["reason"] or "检索预算耗尽"
            st.final_evidence = st.candidates[: self._fetch_k]
            answer = self._agent._abstain_response(st.original_query, st.abstain_reason)
        gs["final"] = {
            "state": st,
            "answer": answer,
            "sources": st.final_evidence,
            "route": st.route,
            "iterations": st.iteration,
            "abstained": st.decision == "ABSTAIN",
        }
        return gs

    # ══════════════════════════════════════════════════
    #  图构建
    # ══════════════════════════════════════════════════

    def _build_graph(self):
        g = StateGraph(GraphState)

        g.add_node("retrieve", self._retrieve_node)
        g.add_node("decompose", self._decompose_node)
        g.add_node("policy", self._policy_node)
        g.add_node("finalize", self._finalize_node)

        # v2.1 cost-aware：precheck（结构预检）→ 按需门控 grader（policy 内部）
        # v2：LLM grader 常开 → retrieve → evaluate → policy
        is_cost_aware = hasattr(self._agent, "_decide_once")
        if is_cost_aware:
            g.add_node("precheck", self._precheck_node)
            g.add_edge(START, "precheck")
            # B2：可靠多问号 → 直接 decompose（跳过全问题初始检索）
            g.add_conditional_edges(
                "precheck",
                lambda gs: "decompose" if gs.get("decision") == "DECOMPOSE" else "retrieve",
                {"decompose": "decompose", "retrieve": "retrieve"},
            )
            g.add_edge("decompose", "policy")
            g.add_edge("retrieve", "policy")
        else:
            g.add_node("evaluate", self._evaluate_node)
            g.add_edge(START, "retrieve")
            g.add_edge("retrieve", "evaluate")
            g.add_edge("evaluate", "policy")
            # DECOMPOSE 执行后回到评估（plan 执行完 → completeness check）
            g.add_edge("decompose", "evaluate")

        # conditional edge：policy 决策 → 路由（与 while 循环语义一致）
        g.add_conditional_edges(
            "policy",
            self._route,
            {
                "ACCEPT": "finalize",
                "RETRIEVE": "retrieve",
                "DECOMPOSE": "decompose",
                "ABSTAIN": "finalize",
            },
        )

        # 终局（finalize 无后续边 → 隐式 END）
        g.add_edge("finalize", END)
        return g.compile()

    @staticmethod
    def _route(gs: GraphState) -> Literal["ACCEPT", "RETRIEVE", "DECOMPOSE", "ABSTAIN"]:
        """路由条件：预算耗尽时强制 ABSTAIN（与自定义 runner 的 while 条件一致）"""
        st = gs["state"]
        if gs["decision"] in ("RETRIEVE", "DECOMPOSE") and st.retrieval_budget <= 0:
            return "ABSTAIN"
        # policy 保证 decision ∈ 4 种动作之一
        return cast(Literal["ACCEPT", "RETRIEVE", "DECOMPOSE", "ABSTAIN"], gs["decision"] or "ABSTAIN")

    # ══════════════════════════════════════════════════
    #  运行入口（与 AgenticRAG.run 相同签名，可无缝替换）
    # ══════════════════════════════════════════════════

    def run(self, question: str, fetch_k: int = 20, verbose: bool = False) -> dict[str, Any]:
        """执行 Agentic RAG（LangGraph runtime）——与自定义 runner 同一契约"""
        from .agentic_rag import AgentState

        self._fetch_k = fetch_k
        st = AgentState(original_query=question)
        st.retrieval_budget = self._agent.max_iterations + 2
        if not hasattr(self._agent, "_decide_once"):
            # v2：无 precheck 节点，初始 route 在此记录（与自定义 runner 一致）；
            # v2.1 由 precheck 节点按预检结果记录 RETRIEVE/DECOMPOSE
            st.route.append("RETRIEVE")

        init: GraphState = {
            "state": st,
            "decision": "",
            "action_mode": "",
            "reason": "",
            "final": None,
            "_grade": {"decision": "insufficient", "reason": "initial", "evidence_score": 0.0},
            "_target_hop": None,
            "_grader_called": False,
        }
        out = self._graph.invoke(init)
        result = out["final"]
        result["elapsed"] = 0.0  # 保持返回契约（parity 对比用 state/answer/route）
        result["grader_called"] = bool(out.get("_grader_called"))
        return result

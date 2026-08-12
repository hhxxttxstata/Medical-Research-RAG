"""
Agentic RAG v2.1 — Cost-aware Agentic Policy（Step 14）

研究问题：能否保持 v2 的 Agentic capability，同时显著减少昂贵模型调用和延迟？

设计来源（Ablation → Failure case → Gate，与项目整体方法论一致）:
  Step 11:  −Grader ≈ full          → LLM grader 在很多时候信号冗余
  Step 12:  bh_ood_02               → 但 reranker 高相关(0.946) ≠ 答案被支持，
                                       信号冲突时 grader 不可删
  结论：不是删除 Grader，而是**不要 Always-on Grader** —— 只在 uncertainty / conflict 时调用。

v2.1 相对 v2 的改动:
  1. Cheap Evidence Signals（零 LLM 成本）先行：
       - Completeness（hop 级支持状态）
       - Hop Support（reranker 分数，非 LLM）
       - top1 Relevance（reranker）
       - Lexical overlap / entity overlap（规则）
  2. Signal Gate：
       - clearly supported  → ACCEPT（不调 LLM grader / policy）
       - clearly missing    → targeted RETRIEVE（不调 LLM grader / policy）
       - uncertain / conflict → 才调用 LLM Grader → LLM Policy
  3. Observability（生产可监控）：
       每题记录 policy_source / grader_called / grader_reason / fallback_used /
       operational_error / retrieval_calls / grader_calls / generation_calls /
       iterations / latency_ms

用例锚点：
  - CONFLICT 典型：reranker very high + completeness says unsupported
    （bh_ood_02 模式）→ grader 必须被调用
  - clearly supported：top1 ≥ 0.5 且（无 plan 或 completeness == 1.0）
  - clearly missing：top1 < 0.05 且无 plan
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from .agentic_rag import (
    _GENERIC_TERMS,
    GRADER_SYSTEM_PROMPT,
    GRADER_USER_PROMPT,
    POLICY_SYSTEM_PROMPT,
    POLICY_USER_PROMPT,
    AgentState,
    HopState,
    _is_out_of_domain,
)
from .generator import LLMGenerator, compute_relevance, create_generator
from .retriever import Retriever

# 便宜信号阈值（与 v2 policy 一致，冻结）
SUPPORTED_TOP1 = 0.5  # reranker top1 ≥ 0.5 → SUPPORTED
CLEARLY_IRRELEVANT = 0.05  # top1 < 0.05 → clearly missing（无 plan 时）

# 语言模型调用类型（供 observability 分类）
CALL_GRADER = "grader"
CALL_DECOMPOSE = "decompose"
CALL_POLICY = "policy"
CALL_GENERATION = "generation"


@dataclass
class CostObservation:
    """Step 14 Observability：单次 run 的成本/路由观测记录

    面试话术对应："Policy 会显式记录 fallback source 和 error type；
    单独监控 grader fallback rate、model timeout rate 和 route distribution，
    防止能力静默退化。"
    """

    policy_source: str = ""  # cheap_signal / llm / rule_fallback
    grader_called: bool = False
    grader_reason: str = ""  # uncertainty / conflict / forced
    fallback_used: bool = False
    operational_error: str = "none"  # none / timeout / api_error
    retrieval_calls: int = 0
    grader_calls: int = 0
    generation_calls: int = 0
    policy_llm_calls: int = 0
    iterations: int = 0
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_source": self.policy_source,
            "grader_called": self.grader_called,
            "grader_reason": self.grader_reason,
            "fallback_used": self.fallback_used,
            "operational_error": self.operational_error,
            "retrieval_calls": self.retrieval_calls,
            "grader_calls": self.grader_calls,
            "generation_calls": self.generation_calls,
            "policy_llm_calls": self.policy_llm_calls,
            "iterations": self.iterations,
            "latency_ms": self.latency_ms,
        }


class CostAwareAgenticRAG:
    """Agentic RAG v2.1 — 与 v2 相同的能力，但 LLM 调用由 uncertainty gate 门控"""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator | None = None,
        reranker=None,
        max_iterations: int = 2,
        grade_temperature: float = 0.0,
    ):
        self.retriever = retriever
        self.generator = generator or create_generator()
        self.reranker = reranker
        self.max_iterations = max_iterations
        self.grade_temperature = grade_temperature
        self._use_llm_grader = True  # uncertainty 时调用 LLM grader
        self._use_llm_policy = True  # grader 仍 uncertain 时调用 LLM policy

    # ══════════════════════════════════════════════════
    #  便宜信号（零 LLM 成本）
    # ══════════════════════════════════════════════════

    def _top1_rel(self, query: str, chunks: list[dict]) -> float:
        """reranker top1 相关性（Cheap Evidence Signal #1）"""
        if not chunks:
            return 0.0
        reranker = self.reranker
        if reranker is not None and getattr(reranker, "model_ready", False):
            try:
                ranked = reranker.rerank(query, list(chunks[:20]), 3)
                if ranked:
                    return float(ranked[0].get("_rerank_score", ranked[0].get("score", 0.0)))
            except Exception:
                pass
        # 无 reranker：退化为向量分
        return float(chunks[0].get("_vector_score", chunks[0].get("score", 0.0)))

    def _lexical_overlap(self, question: str, chunks: list[dict]) -> float:
        """Cheap Evidence Signal #2：区分性实体词面重叠（0-1）

        不用 _has_shared_token（中文 2-gram 太宽松，"如何/实现"等通用 bigram
        几乎在所有中文文本出现 → overlap 恒为 1，CONFLICT 检测失效）。

        只用区分性实体：英文 token（≥3 字符）+ 中文 2-4 字片段（排除通用词，
        复用 v2 _entity_overlap 的实体定义）。无区分性实体时返回 1.0（不误报）。
        """
        if not chunks:
            return 0.0
        cand_text = " ".join(c["text"][:800] for c in chunks[:3]).lower()
        entities = set()
        for tok in re.findall(r"[a-z][a-z0-9\-]{2,}", question.lower()):
            entities.add(tok)
        for n in (4, 3, 2):
            for i in range(len(question) - n + 1):
                frag = question[i : i + n]
                if not re.fullmatch(r"[一-鿿]+", frag):
                    continue
                if frag in _GENERIC_TERMS:
                    continue
                entities.add(frag)
        if not entities:
            return 1.0
        hit = sum(1 for e in entities if e in cand_text)
        return hit / len(entities)

    def _completeness(self, state: AgentState) -> float:
        """Cheap Evidence Signal #3：hop 级支持完成度（复用 v2 定义）"""
        if not state.hops:
            return self._top1_rel(state.original_query, state.evidence_bank) >= SUPPORTED_TOP1 and 1.0 or 0.0
        required = [h for h in state.hops if h.required]
        if not required:
            return 0.0
        supported = 0
        for hop in required:
            score = hop.evidence_score
            hop.support_status = "SUPPORTED" if score >= 0.5 else ("PARTIAL" if score >= 0.05 else "MISSING")
            if hop.support_status == "SUPPORTED":
                supported += 1
        return supported / len(required)

    def _signal_verdict(self, state: AgentState, question: str) -> tuple[str, str]:
        """Cheap Signal Gate：只用便宜信号给出（verdict, reason）

        Returns:
            (verdict, reason)
              verdict ∈ {ACCEPT, RETRIEVE, ABSTAIN, DECOMPOSE, UNCERTAIN}
            UNCERTAIN 表示需要 LLM 介入（调用 grader → policy）。
        """
        top1 = self._top1_rel(question, state.evidence_bank)
        comp = self._completeness(state)
        ood_rule = _is_out_of_domain(question)

        # ── 0. multi-part 结构信号（Step 14 修正，零成本）：先拆解再评估 ──
        # （bh_multi_01 / bh_partial_01 实证：cheap gate 直接 ACCEPT 单跳 → 丢 hop，
        #   正确行为是 DECOMPOSE → hop 级证据采集。拆解是 LLM 调用，但该调用
        #   换取 hop 级完整性，比盲目 ACCEPT 更划算。）
        from .agentic_rag import AgenticRAG

        is_multi_part = AgenticRAG._is_multi_part
        if is_multi_part(question) and not state.hops and not state.decompose_attempted:
            return "DECOMPOSE", "问题含多个独立子问题，先生成 hop plan"

        # ── 0b. 时间敏感冲突（bh_ood_02 模式，零成本信号）：问题含年份 → UNCERTAIN ──
        # （v2 实证：reranker top1=0.946 + 词面有"肺栓塞/指南"重叠仍可能 OOD——
        #   "2026 年…新指南"在知识库里没有答案。时间敏感问题不能靠相关性
        #   ACCEPT（相关性高 ≠ 答案存在），必须交给 grader 裁决答案是否被支持。）
        if re.search(r"(?:20|19)\d{2}\s*年", question):
            return "UNCERTAIN", "问题含年份（时间敏感），需 grader 裁决答案是否被支持"

        # ── clearly supported 1：有 plan 且全部 required hop SUPPORTED ──
        # （v2 的 bh_multi_01 修正：原始问题混合多子问题，rerank top1 可能只命中其一，
        #   但 hop 级证据已齐 → ACCEPT，无需 LLM）
        if state.hops and comp >= 1.0:
            return "ACCEPT", f"全部 hop SUPPORTED（comp={comp:.2f}）"

        # ── clearly supported 2：无 plan 且 top1 高 ──
        if not state.hops and top1 >= SUPPORTED_TOP1:
            # CONFLICT 检查（bh_ood_02 模式）：top1 高但词面无重叠 → 可能是语义近邻
            # （reranker very high + support says unsupported）→ 必须调 grader
            if ood_rule or self._lexical_overlap(question, state.evidence_bank) <= 0.0:
                return (
                    "UNCERTAIN",
                    f"top1={top1:.2f} 但词面无重叠（{ood_rule and 'OOD规则' or '疑似近邻'}），需 grader 裁决",
                )
            return "ACCEPT", f"top1={top1:.2f} 且证据完整（comp={comp:.2f}）"

        # ── clearly missing：无相关证据 ──
        if top1 < CLEARLY_IRRELEVANT and not state.hops:
            if state.retrieval_budget > 0 and state.iteration < self.max_iterations:
                return "RETRIEVE", f"top1={top1:.3f} 无相关证据，targeted 再试一次"
            return "ABSTAIN", f"top1={top1:.3f} 证据与问题无关且预算耗尽"

        # ── clearly incomplete（有 plan）：hop 有 MISSING → targeted retrieve ──
        if state.hops and comp < 1.0:
            missing = next((h for h in state.hops if h.required and h.support_status == "MISSING"), None)
            if missing is not None and state.retrieval_budget > 0:
                return "RETRIEVE", f"hop_{missing.hop_id} 证据缺失（{missing.subquery[:30]}）"
            if missing is None:
                # 全部 PARTIAL：需要 LLM 裁决是否足以作答
                return "UNCERTAIN", f"hop 全 PARTIAL（comp={comp:.2f}），需 grader 裁决"
            return "ABSTAIN", f"hop_{missing.hop_id} 证据缺失且预算耗尽"

        # ── 中间带 → UNCERTAIN（调用 LLM）──
        return "UNCERTAIN", f"top1={top1:.2f} comp={comp:.2f} 处于中间带"

    # ══════════════════════════════════════════════════
    #  LLM 调用（仅 UNCERTAIN 时触发）
    # ══════════════════════════════════════════════════

    def evidence_grade(self, question: str, chunks: list[dict]) -> dict:
        """LLM Grader（与 v2 相同 prompt；失败降级规则判定并标记 fallback）"""
        if not chunks:
            return {
                "decision": "insufficient",
                "reason": "检索结果为空",
                "evidence_score": 0.0,
                "mode": "rule",
                "fallback_used": True,
            }
        try:
            chunk_text = "\n\n".join(f"[{i + 1}] {c['text'][:400]}" for i, c in enumerate(chunks[:12]))
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": GRADER_USER_PROMPT.format(question=question, chunks=chunk_text)},
                ],
                temperature=self.grade_temperature,
                max_tokens=256,
                call_type=CALL_GRADER,
            )
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                decision = data.get("decision", "insufficient")
                if decision not in ("sufficient", "insufficient", "needs_decomposition"):
                    decision = "insufficient"
                return {
                    "decision": decision,
                    "reason": str(data.get("reason", "")),
                    "evidence_score": float(data.get("evidence_score", 0.0)),
                    "mode": "llm",
                    "fallback_used": False,
                }
        except Exception:
            pass
        # 规则 fallback（v2 同款）
        if _is_out_of_domain(question):
            return {
                "decision": "insufficient",
                "reason": "问题明显超出医学知识库领域（规则判定）",
                "evidence_score": 0.0,
                "mode": "rule",
                "unsupported": True,
                "fallback_used": True,
            }
        rel = compute_relevance(question, chunks)
        from .agentic_rag import _has_shared_token

        if rel["is_relevant"]:
            has_lexical_overlap = rel["overlap"] > 0.0 or any(
                _has_shared_token(question, c["text"]) for c in chunks[:3]
            )
            if not has_lexical_overlap:
                return {
                    "decision": "insufficient",
                    "reason": "语义分高但词面无重叠，疑似领域外问题（规则判定）",
                    "evidence_score": rel["top1_score"],
                    "mode": "rule",
                    "unsupported": True,
                    "fallback_used": True,
                }
        decision = "sufficient" if rel["is_relevant"] else "insufficient"
        return {
            "decision": decision,
            "reason": rel["reason"],
            "evidence_score": rel["top1_score"],
            "mode": "rule",
            "fallback_used": True,
        }

    def decompose_plan(self, question: str) -> list[dict]:
        """LLM 拆解（v2 同款；失败返回 []）"""
        system = """\
你是一个医学问题规划器。将问题拆解为结构化的证据获取计划（hop plan）。

## 规则
- 只拆真正需要多步证据的问题（对比、流程、多实体、跨文档）
- 简单问题：{"plan": [], "decomposed": false}
- 每个 hop 是一个可独立检索的子问题
- 若 hop B 依赖 hop A 的答案，标记 depends_on

## 输出（严格 JSON）
{"plan": [{"hop_id": 1, "question": "...", "depends_on": null}], "decomposed": true}
"""
        user = f"问题：{question}\n请生成证据获取计划："
        try:
            response = self.generator.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=256,
                call_type=CALL_DECOMPOSE,
            )
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                if data.get("decomposed"):
                    plan = []
                    for item in data.get("plan", []):
                        plan.append(
                            {
                                "hop_id": int(item.get("hop_id", len(plan) + 1)),
                                "question": str(item.get("question", "")).strip(),
                                "depends_on": item.get("depends_on"),
                                "status": "PENDING",
                            }
                        )
                    if plan and all(p["question"] for p in plan):
                        return plan
        except Exception:
            pass
        return []

    def policy_llm(self, question: str, state: AgentState, grade: dict) -> tuple[str, str, bool]:
        """LLM Policy（v2 同款；失败返回 (None, None, False)）"""
        from .agentic_rag import AgenticRAG

        entity_overlap = AgenticRAG._entity_overlap

        try:
            chunks_text = "\n\n".join(f"[{i + 1}] {c['text'][:300]}" for i, c in enumerate(state.candidates[:8]))
            history_text = "; ".join(h.get("query", "")[:60] for h in state.retrieval_history[-3:]) or "（无）"
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": POLICY_USER_PROMPT.format(
                            question=question,
                            n=min(len(state.candidates), 8),
                            chunks=chunks_text,
                            status=grade.get("decision", "insufficient"),
                            score=round(grade.get("evidence_score", 0.0), 2),
                            top1_rel=round(self._top1_rel(question, state.candidates), 3),
                            entity_overlap=round(entity_overlap(question, state.candidates), 2),
                            history=history_text,
                            iteration=state.iteration,
                            max_iterations=self.max_iterations,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=128,
                call_type=CALL_POLICY,
            )
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                action = str(data.get("action", "")).upper()
                if action not in ("ACCEPT", "RETRIEVE", "DECOMPOSE", "ABSTAIN"):
                    action = "RETRIEVE"
                return action, str(data.get("reason", "")), False
        except Exception:
            pass
        return None, None, True  # fallback 标记

    # ══════════════════════════════════════════════════
    #  主循环
    # ══════════════════════════════════════════════════

    def run(self, question: str, fetch_k: int = 20, verbose: bool = True) -> dict[str, Any]:
        obs = CostObservation()
        state = AgentState(original_query=question)
        state.retrieval_budget = self.max_iterations + 2
        t0 = time.time()
        obs.retrieval_calls += 1

        # ── Iteration 0：初始检索 ──
        state.route.append("RETRIEVE")
        initial = self.retriever._hybrid_retrieve(question, fetch_k=fetch_k)
        state.retrieval_history.append(
            {"query": question, "sources": initial, "iteration": state.iteration, "reason": "initial"}
        )
        self._accumulate(state, initial)
        state.iteration += 1
        state.retrieval_budget -= 1

        grade: dict = {"decision": "insufficient", "reason": "initial", "evidence_score": 0.0}
        decision, action_mode = self._decide_once(question, state, grade, obs, fetch_k, verbose)
        obs.policy_source = action_mode or "cheap_signal"

        while decision in ("RETRIEVE", "DECOMPOSE") and state.retrieval_budget > 0:
            state.route.append(decision)
            if verbose:
                print(f"  🤖 [{state.iteration}] {decision}: {grade.get('reason', '')[:60]}")

            if decision == "DECOMPOSE":
                state.decompose_attempted = True
                obs.grader_reason = "forced"  # multi-part 触发拆解
                plan = self.decompose_plan(question)
                obs.policy_source = "llm"
                if verbose:
                    print(f"      🔗 plan: {plan}")
                if plan:
                    state.plan = plan
                    state.hops = [
                        HopState(
                            hop_id=p["hop_id"], subquery=p["question"], required=True, depends_on=p.get("depends_on")
                        )
                        for p in plan
                    ]
                    state.evidence_by_hop = {h.hop_id: [] for h in state.hops}
                    for hop in state.hops:
                        if state.retrieval_budget <= 0:
                            break
                        self._targeted_retrieve(state, hop, fetch_k)
                        obs.retrieval_calls += 1
                        state.iteration += 1
                        state.retrieval_budget -= 1
                    if missing := self._missing_hops(state):
                        for hop in missing:
                            if state.retrieval_budget <= 0:
                                break
                            self._targeted_retrieve(state, hop, fetch_k)
                            obs.retrieval_calls += 1
                            state.iteration += 1
                            state.retrieval_budget -= 1
                else:
                    state.hops = []
                    state.plan = []
                    more = self.retriever._hybrid_retrieve(question, fetch_k=fetch_k)
                    state.retrieval_history.append(
                        {"query": question, "sources": more, "iteration": state.iteration, "reason": "retry"}
                    )
                    self._accumulate(state, more)
                    obs.retrieval_calls += 1
                    state.iteration += 1
                    state.retrieval_budget -= 1
            else:  # RETRIEVE
                target = grade.get("target_hop")
                if target and state.hops:
                    hop = next((h for h in state.hops if h.hop_id == target["hop_id"]), None)
                    if hop:
                        self._targeted_retrieve(state, hop, fetch_k)
                        obs.retrieval_calls += 1
                    else:
                        more = self.retriever._hybrid_retrieve(target.get("query", question), fetch_k=fetch_k)
                        state.retrieval_history.append(
                            {
                                "query": target.get("query", question),
                                "sources": more,
                                "iteration": state.iteration,
                                "reason": "targeted",
                            }
                        )
                        self._accumulate(state, more)
                        obs.retrieval_calls += 1
                else:
                    new_query = self._build_retrieval_variant(state, question)
                    if verbose:
                        print(f"      🔎 再检索: {new_query[:60]}")
                    more = self.retriever._hybrid_retrieve(new_query, fetch_k=fetch_k)
                    state.retrieval_history.append(
                        {"query": new_query, "sources": more, "iteration": state.iteration, "reason": "retrieve"}
                    )
                    self._accumulate(state, more)
                    obs.retrieval_calls += 1
                state.iteration += 1
                state.retrieval_budget -= 1

            decision, action_mode = self._decide_once(question, state, grade, obs, fetch_k, verbose)
        obs.policy_source = action_mode or "cheap_signal"

        # ── 终局决策 ──
        if decision == "ACCEPT":
            state.decision = "ACCEPT"
            state.route.append("ACCEPT")
            if verbose:
                print(f"  ✅ ACCEPT (comp={state.completeness:.2f})")
        elif decision == "ABSTAIN" or state.retrieval_budget <= 0:
            state.decision = "ABSTAIN"
            state.route.append("ABSTAIN")
            state.abstain_reason = grade.get("reason", "检索预算耗尽")
            if verbose:
                print(f"  🚫 ABSTAIN: {state.abstain_reason[:60]}")
        else:
            state.decision = "ABSTAIN"
            state.route.append("ABSTAIN")
            state.abstain_reason = grade.get("reason", "证据不足")

        # ── 生成 ──
        if state.decision == "ACCEPT":
            state.final_evidence = self._select_final_evidence(question, state.candidates, fetch_k, state)
            answer, op_err = self._generate(question, state.final_evidence)
            obs.generation_calls = 1
            obs.operational_error = op_err
        else:
            state.final_evidence = state.candidates[:fetch_k]
            answer = self._abstain_response(question, state.abstain_reason)

        obs.iterations = state.iteration
        obs.latency_ms = int((time.time() - t0) * 1000)
        if obs.operational_error != "none":
            obs.fallback_used = True

        return {
            "state": state,
            "answer": answer,
            "sources": state.final_evidence,
            "route": state.route,
            "iterations": state.iteration,
            "abstained": state.decision == "ABSTAIN",
            "elapsed": round(time.time() - t0, 2),
            "observation": obs.to_dict(),
        }

    # ══════════════════════════════════════════════════
    #  决策骨架（Gate 优先，LLM 兜底）
    # ══════════════════════════════════════════════════

    def _decide_once(
        self, question: str, state: AgentState, grade: dict, obs: CostObservation, fetch_k: int, verbose: bool
    ) -> tuple[str, str]:
        """单轮决策：Cheap Signal Gate → (UNCERTAIN 时) LLM Grader → LLM Policy

        Returns:
            (decision, action_mode)  action_mode ∈ {cheap_signal, llm, rule_fallback}
        """
        from .agentic_rag import AgenticRAG

        is_multi_part = AgenticRAG._is_multi_part

        # ── 1. Cheap Signal Gate ──
        verdict, reason = self._signal_verdict(state, question)
        if verdict != "UNCERTAIN":
            grade["reason"] = reason
            state.evidence_score = self._top1_rel(question, state.evidence_bank)
            if verdict == "RETRIEVE" and state.hops:
                missing = self._missing_hops(state)
                if missing:
                    grade["target_hop"] = {"hop_id": missing[0].hop_id, "query": missing[0].subquery}
            return verdict, "cheap_signal"

        # ── 2. UNCERTAIN → LLM Grader（或 fallback）──
        obs.grader_called = True
        obs.grader_reason = self._classify_uncertainty(reason)
        grade = self.evidence_grade(question, state.candidates)
        state.evidence_score = grade["evidence_score"]
        if grade.get("fallback_used"):
            obs.fallback_used = True
        g_decision = grade.get("decision", "insufficient")
        g_reason = grade.get("reason", "")

        # grader 直接给出明确结论 → 用 grader 结论（v2 同款映射，无需 LLM policy）
        if g_decision == "sufficient":
            grade["reason"] = g_reason or "证据充分"
            return "ACCEPT", "llm"
        if g_decision == "needs_decomposition":
            grade["reason"] = g_reason or "需要拆解"
            return "DECOMPOSE", "llm"
        # unsupported / insufficient 的区分
        unsupported = grade.get("unsupported") or (
            g_decision == "insufficient"
            and any(k in g_reason for k in ("未提及", "未找到", "不存在", "没有提及", "无相关"))
        )
        if unsupported and not (is_multi_part(question) and not state.hops):
            if state.retrieval_budget > 0 and state.iteration < self.max_iterations:
                grade["reason"] = "证据不支撑答案（grader），targeted 再试一次"
                return "RETRIEVE", "llm"
            grade["reason"] = g_reason or "证据不支撑答案"
            return "ABSTAIN", "llm"

        # multi-part 未拆过 → 拆解
        if is_multi_part(question) and not state.hops and not state.decompose_attempted:
            grade["reason"] = "问题含多个独立子问题，生成结构化 plan"
            return "DECOMPOSE", "llm"

        # ── 3. 仍不确定 → LLM Policy ──
        if self._use_llm_policy:
            action, reason_llm, fb = self.policy_llm(question, state, grade)
            if action is not None:
                grade["reason"] = reason_llm
                return action, "llm"
            obs.fallback_used = True

        # ── 4. 规则兜底（grader/policy 都失败）──
        obs.fallback_used = True
        obs.policy_source = "rule_fallback"
        if g_decision == "sufficient":
            return "ACCEPT", "rule_fallback"
        if state.retrieval_budget > 0 and state.iteration < self.max_iterations:
            return "RETRIEVE", "rule_fallback"
        return "ABSTAIN", "rule_fallback"

    @staticmethod
    def _classify_uncertainty(reason: str) -> str:
        """把 cheap gate 的 reason 分类为 grader_reason（uncertainty/conflict/forced）"""
        if "词面无重叠" in reason or "CONFLICT" in reason:
            return "conflict"
        if "中间带" in reason or "PARTIAL" in reason:
            return "uncertainty"
        return "uncertainty"

    # ══════════════════════════════════════════════════
    #  辅助（与 v2 行为一致）
    # ══════════════════════════════════════════════════

    def _accumulate(self, state: AgentState, new_chunks: list[dict]) -> None:
        seen = {c["id"] for c in state.evidence_bank}
        for c in new_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                state.evidence_bank.append(c)
        seen2 = {c["id"] for c in state.candidates}
        for c in new_chunks:
            if c["id"] not in seen2:
                seen2.add(c["id"])
                state.candidates.append(c)
        self._assign_to_hops(state)

    def _assign_to_hops(self, state: AgentState) -> None:
        if not state.hops:
            return
        state.evidence_by_hop = {}
        for hop in state.hops:
            state.evidence_by_hop[hop.hop_id] = []
        reranker = self.reranker
        if reranker is not None and getattr(reranker, "model_ready", False):
            try:
                for hop in state.hops:
                    ranked = reranker.rerank(hop.subquery, list(state.evidence_bank), 3)
                    state.evidence_by_hop[hop.hop_id] = ranked
                    hop.evidence_ids = [c["id"] for c in ranked]
                    hop.evidence_score = ranked[0].get("_rerank_score", 0.0) if ranked else 0.0
                return
            except Exception:
                pass
        for hop in state.hops:
            hop_ev = [c for c in state.evidence_bank[:10] if self._shared(hop.subquery, c["text"])]
            state.evidence_by_hop[hop.hop_id] = hop_ev[:3]
            hop.evidence_ids = [c["id"] for c in hop_ev[:3]]

    @staticmethod
    def _shared(query: str, text: str) -> bool:
        from .agentic_rag import _has_shared_token

        return _has_shared_token(query, text)

    def _targeted_retrieve(self, state: AgentState, hop: HopState, fetch_k: int) -> None:
        hop.retrieval_attempts += 1
        results = self.retriever._hybrid_retrieve(hop.subquery, fetch_k=fetch_k)
        state.retrieval_history.append(
            {"query": hop.subquery, "sources": results, "iteration": state.iteration, "reason": "targeted-hop"}
        )
        self._accumulate(state, results)

    def _missing_hops(self, state: AgentState) -> list[HopState]:
        return [h for h in state.hops if h.required and h.support_status == "MISSING"]

    def _build_retrieval_variant(self, state: AgentState, question: str) -> str:
        from collections import Counter

        freq: Counter = Counter()
        for c in state.candidates[:10]:
            text = c["text"]
            for m in re.findall(r"[一-鿿]{2,4}", text):
                if m not in question:
                    freq[m] += 1
            for m in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,20}", text):
                if m.lower() not in question.lower():
                    freq[m] += 1
        top_terms = [w for w, _ in freq.most_common(4) if freq[w] >= 2][:4]
        if top_terms:
            return f"{question} {' '.join(top_terms)}"
        return question

    def _select_final_evidence(self, question: str, candidates: list[dict], k: int, state: AgentState) -> list[dict]:
        if state.hops:
            merged: list[dict] = []
            seen = set()
            for hop in state.hops:
                for c in state.evidence_by_hop.get(hop.hop_id, []):
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        merged.append(c)
            if merged:
                return merged[:k]
        reranker = self.reranker
        if reranker is not None and getattr(reranker, "model_ready", False):
            try:
                return reranker.rerank(question, list(candidates), k)
            except Exception:
                pass
        return candidates[:k]

    def _generate(self, question: str, chunks: list[dict]) -> tuple[str, str]:
        """生成回答；区分 operational failure（timeout/model error）与 epistemic abstain"""
        from .generator import build_rag_prompt

        if not chunks:
            return self._abstain_response(question), "none"
        try:
            messages, source_map, relevance = build_rag_prompt(question, chunks)
            gen = self.generator.generate_structured((messages, source_map, relevance), self_reflect=False)
            if gen["valid"]:
                return gen["raw"], "none"
            return gen["raw"], "none"
        except Exception:
            # Operational failure（API timeout / network）→ 不降级为 ABSTAIN，单独标记
            return self._abstain_response(question, "生成阶段 API 调用失败（operational error）"), "api_error"

    @staticmethod
    def _abstain_response(question: str, reason: str = "") -> str:
        parts = [
            "**结论：**",
            f"知识库中未找到足够证据回答：{question}",
            "",
            "**依据：**",
            "检索到的内容与问题相关性不足，无法给出可靠回答。",
        ]
        if reason:
            parts.extend(["", f"> 📊 拒答原因: {reason}"])
        parts.extend(
            [
                "",
                "**建议下一步：**",
                "1. 换一种表述方式提问，或补充更多细节",
                "2. 联系管理员确认该主题是否已在知识库中",
            ]
        )
        return "\n".join(parts)

"""
Agentic RAG v1 — 动态检索代理（Step 10）

核心思想：检索不是固定链路，而是 Agent 根据证据质量动态决策的过程。

State:
    original_query   原始问题（不可变）
    retrieval_history  [{query, sources, iteration, reason}]
    candidates        已检索候选 chunk（去重累积）
    evidence_score    当前证据充分性评分 (0-1)
    route             决策路径（accept/retrieve/decompose/abstain 序列）
    iteration         当前迭代数（max_iterations=2）
    final_evidence    最终选定的证据集

工具（4 个）:
    hybrid_search     混合检索（向量 + BM25 → RRF）
    decompose         把复杂问题拆成子问题（multi-hop 专用）
    evidence_grade    评估当前证据是否充分（LLM grader）
    generate          用最终证据生成回答

决策（4 种）:
    ACCEPT    证据充分 → 进入生成
    RETRIEVE  证据不足 → 换角度再检索
    DECOMPOSE multi-hop → 拆子问题后检索
    ABSTAIN   证据不足且已尽力 → 拒答/说明证据不足

约束:
    max_iterations = 2（RETRIEVE/DECOMPOSE 总迭代上限）
    第一版不做 Rewrite（Step 1–7 已证明无正收益）
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .generator import LLMGenerator, compute_relevance, create_generator
from .retriever import Retriever

# ── 决策提示词 ─────────────────────────────────────────

GRADER_SYSTEM_PROMPT = """\
你是一个 RAG 证据评估器。给定用户问题和检索到的证据片段，判断证据是否足以回答该问题。

## 判定标准
- sufficient: 证据包含回答所需的关键信息（事实、数值、定义、机制、步骤等），
  可基于证据给出合理回答。不要求证据完整叙述全文——只要关键答案信息存在即可。
- insufficient: 证据与问题无关，或仅提及主题但缺失所有关键答案信息。
- needs_decomposition: 问题包含多个独立子问题（如"X和Y有什么区别"、"从A到B的完整流程"），
  需要拆解后分别检索才能完整回答

## 输出（严格 JSON，不要其他内容）
{"decision": "sufficient" | "insufficient" | "needs_decomposition", "reason": "判断依据（中文，30字内）", "evidence_score": 0.0-1.0}
"""

GRADER_USER_PROMPT = """## 用户问题
{question}

## 检索到的证据片段
{chunks}

请评估证据充分性："""

POLICY_SYSTEM_PROMPT = """\
你是一个 RAG Agent 的决策模块（Policy Node）。基于当前完整状态，选择下一步动作。

## 输入状态
- 用户问题
- 当前检索到的证据片段（去重累积）
- Evidence Status：SUFFICIENT / INSUFFICIENT / UNSUPPORTED
- Evidence Score：LLM 对证据充分性的 0-1 评分
- 检索历史：已经检索过哪些查询
- 当前迭代数 / 最大迭代数

## 动作空间（4 种）
- ACCEPT：证据已足以回答 → 进入生成
- RETRIEVE：证据不足但问题主题在知识库内，换角度再检索
- DECOMPOSE：问题含多个独立子问题（对比/流程/多实体），拆解后分别检索
- ABSTAIN：知识库不支持（OOD / 无答案），停止并拒答

## 决策原则
- 证据评分 ≥ 0.6 或证据片段明显包含答案 → ACCEPT
- 证据不足但候选与问题主题相关（能找到关键实体）→ RETRIEVE 或 DECOMPOSE
- 证据不足且主题无关 / 领域外 → ABSTAIN
- 已迭代 2 次仍未充分 → ABSTAIN（避免无限循环）
- 注意：chunk 是 300-500 字的片段，不要求单个片段包含完整答案；
  只要关键答案信息（数值/定义/步骤/机制）已出现即可 ACCEPT
- 额外信号会一并给出：
    - top1_relevance：cross-encoder 对 top1 片段与问题的相关性（0-1，越高越相关）
    - entity_overlap：问题中的关键实体在证据中出现的比例（0-1）
    top1_relevance ≥ 0.7 或 entity_overlap ≥ 0.6 → 强烈倾向 ACCEPT

## 输出（严格 JSON，不要其他内容）
{"action": "ACCEPT" | "RETRIEVE" | "DECOMPOSE" | "ABSTAIN", "reason": "决策依据（中文，40字内）"}
"""

POLICY_USER_PROMPT = """## 用户问题
{question}

## 当前证据片段（前 {n} 个）
{chunks}

## Evidence Status
{status}

## Evidence Score
{score}

## top1 相关性（cross-encoder）
{top1_rel}

## 关键实体覆盖率
{entity_overlap}

## 检索历史
{history}

## 当前迭代
{iteration}/{max_iterations}

请选择下一步动作："""

DECOMPOSE_SYSTEM_PROMPT = """\
你是一个医学问题拆解助手。如果一个问题包含多个独立子问题，将其拆成 2-3 个可独立检索的子问题。

## 规则
- 只拆真正需要分步回答的问题（对比、流程、多实体关系）
- 简单问题不要拆，原样返回
- 每个子问题保持完整、可独立检索

## 输出（严格 JSON，不要其他内容）
{"sub_questions": ["子问题1", "子问题2"], "decomposed": true}
若不需要拆解：{"sub_questions": [], "decomposed": false}
"""

DECOMPOSE_USER_PROMPT = """问题：{question}
请拆解："""

# 领域外关键词（规则判定用，与 Retriever._rewrite_gate 的领域判断一致）
_OOD_PATTERNS = [
    r"经济增长|GDP|股票|基金|汇率|期货",  # 经济金融
    r"Kubernetes|RBAC|CloudFormation|goroutine|区块链|PoW|PoS",  # 技术无关领域
    r"量子纠错|量子计算|科举|殿试|会试|非洲猪瘟|日语敬语|电动车锂电池",  # 其他
    r"React|useEffect|useLayoutEffect|async/await|Nginx|异步编程",
]

# 中文通用词（不计入主题相关性判定的区分性实体）
_GENERIC_TERMS = {
    "如何",
    "什么",
    "哪些",
    "多少",
    "为什么",
    "怎样",
    "是否",
    "包括",
    "主要",
    "用于",
    "系统",
    "方法",
    "技术",
    "模型",
    "数据",
    "治疗",
    "诊断",
    "检查",
    "流程",
    "步骤",
    "作用",
    "影响",
    "分析",
    "应用",
    "方案",
    "相关",
    "特点",
    "优势",
    "区别",
    "对比",
    "临床",
    "影像",
    "医学",
    "患者",
    "疾病",
    "功能",
    "评估",
    "结果",
    "风险",
    "标准",
    "心血管",
    "长期",
    "不同",
    "之间",
    "以及",
    "进行",
    "可以",
    "需要",
    "通过",
    "患者的",
    "问题",
    "情况",
    "目的",
    "对象",
    "方式",
    "方面",
    "内容",
    "类型",
    "指标",
}


def _is_out_of_domain(question: str) -> bool:
    """粗略领域判定：命中 OOD 关键词 → True（仅用于 LLM 不可用时的 fallback）"""
    return any(re.search(p, question, re.IGNORECASE) for p in _OOD_PATTERNS)


def _has_shared_token(query: str, text: str) -> bool:
    """query 与 chunk 是否有共享词面 token（中文 2-gram 或英文词）"""
    q = query.lower()
    t = text.lower()
    # 英文词共享
    q_words = set(re.findall(r"[a-z][a-z0-9\-]{2,}", q))
    if q_words & set(re.findall(r"[a-z][a-z0-9\-]{2,}", t)):
        return True
    # 中文 2-gram 共享（过滤纯数字/停用）
    stop = {"一个", "多少", "什么", "如何", "哪些", "为什么", "怎样", "是否", "的是", "用于", "中的", "在肺", "栓塞"}
    q_bigrams = set(q[i : i + 2] for i in range(len(q) - 1)) - stop
    t_bigrams = set(t[i : i + 2] for i in range(len(t) - 1))
    return len(q_bigrams & t_bigrams) >= 2


@dataclass
class HopState:
    """Step 13 v2: Hop 级证据状态

    每个 hop 记录：需要什么证据（subquery）、已获得什么证据（evidence_ids）、
    支持状态（support_status）。这是 v2 与 v1 的本质区别——v1 的 candidates
    是 flat list，v2 显式追踪"每个 hop 的证据是否齐了"。
    """

    hop_id: int
    subquery: str
    required: bool = True
    depends_on: int | None = None
    evidence_ids: list[str] = field(default_factory=list)
    evidence_score: float = 0.0
    support_status: str = "PENDING"  # PENDING / SUPPORTED / PARTIAL / MISSING / CONTRADICTED
    retrieval_attempts: int = 0


@dataclass
class AgentState:
    """Agentic RAG 状态

    语义修正（Step 10.5）：
      - evidence_status 与 decision 分离：
          evidence_status ∈ {SUFFICIENT, INSUFFICIENT, UNSUPPORTED}  (证据状态)
          decision        ∈ {ACCEPT, RETRIEVE, DECOMPOSE, ABSTAIN}   (代理动作)
      - ABSTAIN 只代表最终停止并拒绝回答（不再有 ABSTAIN→ACCEPT 降级）
      - route 记录完整动作序列（含循环内 RETRIEVE/DECOMPOSE）

    Step 13 v2 扩展：
      - plan / hops：structured decomposition plan（hop_id + subquery + status）
      - evidence_bank：累积证据池（去重），evidence_by_hop：按 hop 分配
      - completeness：hop 支持完成度（0-1）
      - retrieval_budget：检索预算（ABSTAIN = 预算耗尽）
    """

    original_query: str = ""
    retrieval_history: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)  # v1 兼容：全量候选
    evidence_bank: list[dict] = field(default_factory=list)  # v2：去重证据池
    evidence_by_hop: dict[int, list[dict]] = field(default_factory=dict)
    plan: list[dict] = field(default_factory=list)  # structured plan（LLM 输出）
    hops: list[HopState] = field(default_factory=list)
    completeness: float = 0.0
    evidence_score: float = 0.0
    evidence_status: str = ""  # SUFFICIENT / INSUFFICIENT / UNSUPPORTED
    route: list[str] = field(default_factory=list)
    iteration: int = 0
    retrieval_budget: int = 4
    final_evidence: list[dict] = field(default_factory=list)
    decision: str = ""  # ACCEPT / RETRIEVE / DECOMPOSE / ABSTAIN
    abstain_reason: str = ""
    decompose_attempted: bool = False  # v2: 已尝试过 DECOMPOSE（防规则误判死循环）


class AgenticRAG:
    """Agentic RAG v1 — 4 工具 + 4 决策 + max_iterations=2"""

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator | None = None,
        reranker=None,
        max_iterations: int = 2,
        grade_threshold: float = 0.6,
        grade_temperature: float = 0.0,
    ):
        self.retriever = retriever
        self.generator = generator or create_generator()
        self.reranker = reranker
        self.max_iterations = max_iterations
        self.grade_threshold = grade_threshold
        self.grade_temperature = grade_temperature

    # ══════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════

    def hybrid_search(self, query: str, fetch_k: int = 20, note: str = "") -> list[dict]:
        """工具 1：混合检索（向量 + BM25 → RRF）"""
        results = self.retriever._hybrid_retrieve(query, fetch_k=fetch_k)
        return results

    def evidence_grade(self, question: str, chunks: list[dict]) -> dict:
        """工具 3：证据充分性评估（LLM grader + 规则 fallback）"""
        # ── 规则预判：无候选 → 必然不足 ──
        if not chunks:
            return {"decision": "insufficient", "reason": "检索结果为空", "evidence_score": 0.0, "mode": "rule"}

        # ── LLM 判定 ──
        try:
            chunk_text = "\n\n".join(f"[{i + 1}] {c['text'][:400]}" for i, c in enumerate(chunks[:12]))
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": GRADER_USER_PROMPT.format(question=question, chunks=chunk_text)},
                ],
                temperature=self.grade_temperature,
                max_tokens=256,
                call_type="grader",
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
                }
        except Exception:
            pass

        # ── 规则 fallback：基于相关性的粗略判定 ──
        # LLM 不可用时，先做领域判定：明显非医学问题直接 insufficient + unsupported
        if _is_out_of_domain(question):
            return {
                "decision": "insufficient",
                "reason": "问题明显超出医学知识库领域（规则判定）",
                "evidence_score": 0.0,
                "mode": "rule",
                "unsupported": True,
            }
        rel = compute_relevance(question, chunks)
        if rel["is_relevant"]:
            # 语义分高但词面零重叠 → 疑似 OOD（如"诺贝尔奖""日语敬语"）
            # 医学检索的跨语言/同义表达有重叠；零重叠说明只是语义空间近邻
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
                }
        decision = "sufficient" if rel["is_relevant"] else "insufficient"
        return {"decision": decision, "reason": rel["reason"], "evidence_score": rel["top1_score"], "mode": "rule"}

    def decompose(self, question: str) -> list[str]:
        """工具 2：问题拆解（multi-hop 专用）"""
        try:
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": DECOMPOSE_USER_PROMPT.format(question=question)},
                ],
                temperature=0.0,
                max_tokens=256,
                call_type="decompose",
            )
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                if data.get("decomposed"):
                    subs = [str(s).strip() for s in data.get("sub_questions", []) if str(s).strip()]
                    if subs:
                        return subs
        except Exception:
            pass
        return []

    def generate(self, question: str, chunks: list[dict]) -> str:
        """工具 4：生成回答（基于最终证据）

        Step 13.5 语义修正：generation API timeout ≠ ABSTAIN。
        Operational failure（timeout/api error）单独标记为 ERROR_TIMEOUT / ERROR_MODEL，
        不计入 OOD Reject，也不计入 False Abstain——由评测端统一统计
        Operational Failure Rate。不再把 operational error 降级成拒答（防止
        "OOD Reject 提高"实际只是 "API 挂了"）。
        """
        from .generator import build_rag_prompt

        if not chunks:
            return self._abstain_response(question)
        try:
            messages, source_map, relevance = build_rag_prompt(question, chunks)
            gen = self.generator.generate_structured((messages, source_map, relevance), self_reflect=False)
            return gen["raw"]
        except Exception:
            # Operational failure：标记后返回错误占位，不再伪装成 ABSTAIN。
            # 评测端通过 [OPERATIONAL_ERROR] 前缀识别并计入 Operational Failure Rate。
            return "[OPERATIONAL_ERROR] 生成阶段 API 调用失败（timeout/network），已单独统计，不计入拒答。"

    # ══════════════════════════════════════════════════
    #  决策循环
    # ══════════════════════════════════════════════════

    def _dedup_accumulate(self, state: AgentState, new_chunks: list[dict]) -> None:
        """把新检索结果累积进 candidates（按 id 去重）"""
        seen = {c["id"] for c in state.candidates}
        for c in new_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                state.candidates.append(c)

    # ══════════════════════════════════════════════════
    #  Step 13 v2: Evidence Accumulator + Hop State
    # ══════════════════════════════════════════════════

    def _accumulate_evidence(self, state: AgentState, new_chunks: list[dict]) -> None:
        """v2: Evidence Accumulator —— 证据进 bank（去重），不覆盖历史

        v1 的问题（13B）：每轮检索后直接累积进 candidates 但缺少 hop 归属；
        v2 把证据累积进 evidence_bank，并重新分配给各 hop。
        """
        seen = {c["id"] for c in state.evidence_bank}
        for c in new_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                state.evidence_bank.append(c)
        # 同步到 candidates（兼容 v1 接口）
        self._dedup_accumulate(state, new_chunks)
        # 重新分配证据到 hop
        self._assign_evidence_to_hops(state)

    def _assign_evidence_to_hops(self, state: AgentState) -> None:
        """把 evidence_bank 中的证据按与 hop subquery 的相关性分配给各 hop"""
        if not state.hops:
            return
        state.evidence_by_hop = {}
        for hop in state.hops:
            state.evidence_by_hop[hop.hop_id] = []
        # 用 reranker 对每个 hop 的 subquery 打分分配
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
        # 无 reranker fallback：按词面重叠分配
        for hop in state.hops:
            hop_ev = []
            for c in state.evidence_bank[:10]:
                if _has_shared_token(hop.subquery, c["text"]):
                    hop_ev.append(c)
            state.evidence_by_hop[hop.hop_id] = hop_ev[:3]
            hop.evidence_ids = [c["id"] for c in hop_ev[:3]]

    def _support_status(self, state: AgentState, hop: HopState) -> str:
        """判定单个 hop 的 support status（13C：Relevance ≠ Support）

        SUPPORTED    : 该 hop 有高相关证据（reranker ≥ 0.5）
        PARTIAL      : 有相关但不足（0.05-0.5）
        MISSING      : 无相关证据（< 0.05）或证据为空
        CONTRADICTED : 证据间冲突（本版不实现，保留枚举）
        """
        if hop.evidence_score >= 0.5:
            return "SUPPORTED"
        if hop.evidence_score >= 0.05:
            return "PARTIAL"
        return "MISSING"

    def _compute_completeness(self, state: AgentState) -> tuple[float, list[HopState]]:
        """Completeness Check：所有 required hop 的支持状态汇总

        Returns:
            (completeness 0-1, missing_hops 列表)
        """
        if not state.hops:
            # 无 plan（单跳问题）→ 用 evidence_bank 的 top1 相关性
            top1 = 0.0
            if self.reranker is not None and getattr(self.reranker, "model_ready", False) and state.evidence_bank:
                try:
                    ranked = self.reranker.rerank(state.original_query, list(state.evidence_bank), 1)
                    top1 = ranked[0].get("_rerank_score", 0.0) if ranked else 0.0
                except Exception:
                    top1 = 0.0
            state.completeness = 1.0 if top1 >= 0.5 else (0.5 if top1 >= 0.05 else 0.0)
            return state.completeness, []

        required = [h for h in state.hops if h.required]
        if not required:
            state.completeness = 0.0
            return 0.0, []
        missing = []
        for hop in required:
            status = self._support_status(state, hop)
            hop.support_status = status
            if status == "MISSING":
                missing.append(hop)
        complete = len(required) - len(missing)
        state.completeness = complete / len(required)
        return state.completeness, missing

    def _build_plan_prompt(self, question: str) -> tuple[str, str]:
        """13E: DECOMPOSE 输出 structured plan（不再是裸 subqueries）"""
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
        return system, user

    def decompose_plan(self, question: str) -> list[dict]:
        """13E: 结构化拆解 —— 输出 plan（hop_id + question + depends_on）"""
        system, user = self._build_plan_prompt(question)
        try:
            response = self.generator.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=256,
                call_type="decompose",
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

    def _init_hops_from_plan(self, state: AgentState, plan: list[dict]) -> None:
        """把 plan 转成 HopState 列表"""
        state.plan = plan
        state.hops = [
            HopState(
                hop_id=p["hop_id"],
                subquery=p["question"],
                required=True,
                depends_on=p.get("depends_on"),
            )
            for p in plan
        ]
        state.evidence_by_hop = {h.hop_id: [] for h in state.hops}

    def _targeted_retrieve(self, state: AgentState, hop: HopState, fetch_k: int = 20) -> list[dict]:
        """13D: Targeted hop retrieval —— 用 hop 的 subquery 精确检索缺失证据"""
        hop.retrieval_attempts += 1
        results = self.hybrid_search(hop.subquery, fetch_k=fetch_k, note=f"targeted:{hop.subquery}")
        state.retrieval_history.append(
            {"query": hop.subquery, "sources": results, "iteration": state.iteration, "reason": "targeted-hop"}
        )
        self._accumulate_evidence(state, results)
        return results

    def _find_missing_hop(self, state: AgentState) -> HopState | None:
        """找到第一个 MISSING 的 required hop（供 targeted retrieve）"""
        for hop in state.hops:
            if hop.required and self._support_status(state, hop) == "MISSING":
                return hop
        return None

    def _decide(self, grade: dict) -> tuple[str, str]:
        """把 Evidence Status 映射为 Agent Action（规则 fallback，Policy Node 不可用时）

        Returns:
            (evidence_status, action)
              SUFFICIENT     → ACCEPT
              INSUFFICIENT   → RETRIEVE（迭代未用尽时）
              UNSUPPORTED    → ABSTAIN（迭代用尽 / 领域外）
              needs_decomposition → DECOMPOSE
        """
        decision = grade["decision"]
        if decision == "sufficient":
            return "SUFFICIENT", "ACCEPT"
        if decision == "needs_decomposition":
            return "INSUFFICIENT", "DECOMPOSE"
        # insufficient / unsupported
        if grade.get("unsupported"):
            return "UNSUPPORTED", "ABSTAIN"
        return "INSUFFICIENT", "RETRIEVE"

    def policy(self, question: str, state: AgentState, grade: dict) -> tuple[str, str, str]:
        """Policy Node v2（Step 13C）：Relevance ≠ Support ≠ Completeness

        ACCEPT 的新定义：
            evidence relevant (top1 ≥ 0.5)
            AND required evidence complete (completeness ≥ 1.0 或单跳 top1 ≥ 0.5)
            AND answer support sufficient (LLM grader 不判 unsupported)

        关键修正（bh_ood_02 实证）：reranker top1 高相关 ≠ 答案被支持。
        LLM grader 判 unsupported 时，即使 top1 ≥ 0.5 也不 ACCEPT。

        信号优先级：
            grader UNSUPPORTED      → RETRIEVE / ABSTAIN（高相关也无效）
            top1 ≥ 0.5 + complete   → ACCEPT
            top1 ≥ 0.5 + incomplete → targeted RETRIEVE missing hop
            top1 < 0.05 + budget    → RETRIEVE
            top1 < 0.05 + 耗尽      → ABSTAIN
            中间带                   → LLM Policy

        Returns:
            (evidence_status, action, mode)
            action ∈ {ACCEPT, RETRIEVE, DECOMPOSE, ABSTAIN}，
            RETRIEVE 时 grade["target_hop"] 携带缺失 hop 的 subquery（13D）。
        """
        status, action = self._decide(grade)

        # ── 客观信号：reranker top1 相关性（仅 reranker 可用时生效）──
        top1_rel = 0.0
        reranker_ready = self.reranker is not None and getattr(self.reranker, "model_ready", False)
        if reranker_ready:
            try:
                ranked = self.reranker.rerank(question, list(state.candidates[:20]), 3)
                if ranked:
                    top1_rel = ranked[0].get("_rerank_score", ranked[0].get("score", 0.0))
            except Exception:
                top1_rel = 0.0

        # ── 13C：Completeness Check（有 plan 时）—— 优先于 unsupported ──
        # （bh_multi_01 实证：plan 执行后 hop 全 SUPPORTED，但 grader 仍说
        #  "缺失关键信息"——此时应以 hop 支持状态为准，而不是 grader 的整体判定）
        if state.hops:
            completeness, missing = self._compute_completeness(state)
            state.completeness = completeness
            if completeness >= 1.0:
                # 全部 hop SUPPORTED → ACCEPT（hop 已各自用 subquery 验证，无需再要求
                # 原始问题的 top1——bh_multi_01 实证：原始问题混合两个子问题，
                # rerank top1 可能只命中其一，但 hop 级证据已齐）
                grade["reason"] = f"全部 hop SUPPORTED（completeness={completeness:.2f}）"
                return "SUFFICIENT", "ACCEPT", "signal"
            if missing:
                # 有缺失 hop → targeted retrieve（13D）
                hop = missing[0]
                grade["target_hop"] = {"hop_id": hop.hop_id, "query": hop.subquery}
                grade["reason"] = f"hop_{hop.hop_id} 证据缺失（{hop.subquery[:40]}）"
                return "INSUFFICIENT", "RETRIEVE", "signal"

        # ── 13C 核心：LLM grader 判 UNSUPPORTED → 高相关也不能 ACCEPT ──
        # （bh_ood_02 实证：top1=0.946 但答案不存在，grader 正确识别被 v1 signal 覆盖）
        # 例外：multi-part 问题的"缺失关键信息"是缺子问题证据（应 DECOMPOSE），
        # 不是答案不存在（bh_multi_01 实证：reason"未提及推理加速框架"）。
        unsupported = grade.get("unsupported") or (
            grade.get("decision") == "insufficient" and self._grader_hints_unsupported(grade)
        )
        if unsupported and not (self._is_multi_part(question) and not state.hops):
            # 时间敏感/领域外：预算未耗尽时给一次 targeted 机会，否则 ABSTAIN
            if state.retrieval_budget > 0 and state.iteration < self.max_iterations:
                grade["reason"] = "证据不支撑答案（grader），targeted 再试一次"
                return "INSUFFICIENT", "RETRIEVE", "signal"
            grade["reason"] = grade.get("reason", "证据不支撑答案")
            return "UNSUPPORTED", "ABSTAIN", "signal"

        # ── 13E：multi-part 问题 → 直接 DECOMPOSE（structured plan）──
        # v1 把 multi-part 交 LLM policy 自由选择，导致 RETRIEVE 而非拆解；
        # v2 结构判断命中（多问号/对比词/并列）→ 直接走 plan 流程。
        # 例外：已尝试过拆解（LLM 认为不需要）→ 不再重复，走单跳路径。
        if self._is_multi_part(question) and not state.hops and not state.decompose_attempted:
            grade["reason"] = "问题含多个独立子问题，生成结构化 plan"
            return "INSUFFICIENT", "DECOMPOSE", "signal"

        # ── 单跳路径（无 plan）──
        # LLM grader 已判 SUFFICIENT → 无条件 ACCEPT（全视野判断）
        if action == "ACCEPT":
            grade["reason"] = grade.get("reason", "证据充分")
            return "SUFFICIENT", "ACCEPT", "rule"

        if reranker_ready:
            # 信号硬规则：top1 高度相关 → ACCEPT（单跳问题）
            if top1_rel >= 0.5 and not self._is_multi_part(question):
                grade["reason"] = f"top1 证据相关性 {top1_rel:.2f}（cross-encoder 高相关）"
                return "SUFFICIENT", "ACCEPT", "signal"
            # 完全无相关证据：预算未耗尽 → RETRIEVE；耗尽 → ABSTAIN
            if top1_rel < 0.05:
                if state.retrieval_budget > 0 and state.iteration < self.max_iterations:
                    grade["reason"] = f"top1 证据相关性 {top1_rel:.3f}，换角度再检索"
                    return "INSUFFICIENT", "RETRIEVE", "signal"
                grade["reason"] = f"top1 证据相关性 {top1_rel:.3f}，证据与问题无关"
                return "UNSUPPORTED", "ABSTAIN", "signal"

        # ── LLM Policy 决策（中间带）──
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
                            top1_rel=round(top1_rel, 3),
                            entity_overlap=round(self._entity_overlap(question, state.candidates), 2),
                            history=history_text,
                            iteration=state.iteration,
                            max_iterations=self.max_iterations,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=128,
                call_type="policy",
            )
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                action = str(data.get("action", "")).upper()
                if action not in ("ACCEPT", "RETRIEVE", "DECOMPOSE", "ABSTAIN"):
                    action = "RETRIEVE"  # 非法输出 → 保守：再检索
                reason = str(data.get("reason", ""))
                # status 以 LLM policy 的动作为准重新推导
                if action == "ACCEPT":
                    status = "SUFFICIENT"
                elif action == "ABSTAIN":
                    status = grade.get("unsupported") and "UNSUPPORTED" or "INSUFFICIENT"
                else:
                    status = "INSUFFICIENT"
                grade["reason"] = reason
                return status, action, "policy_llm"
        except Exception:
            pass

        return status, action, "rule"

    @staticmethod
    def _grader_hints_unsupported(grade: dict) -> bool:
        """grader reason 是否提示 unsupported（时间敏感/不存在/未提及）

        bh_ood_02 实证：grader 说"未提及2026年ESC年会"，reason 含"未提及/不存在/没有"。
        仅当 reason 明确指向"答案不存在"才视为 unsupported，避免过度拒答。
        """
        reason = str(grade.get("reason", ""))
        return any(k in reason for k in ("未提及", "未找到", "不存在", "没有提及", "无相关"))

    @staticmethod
    def _is_multi_part(question: str) -> bool:
        """结构判断：问题是否含多个独立子问题（需要分步检索）

        v2 修正（bh_easy_03/hard_02/hard_03 实证）："X和Y分别"不一定是 multi-part——
        同一主题的并列属性（"窗宽和窗位"）单跳即可答；词法判断"共享实体"过于脆弱
        （"外部验证集"在分句间共享但问题是两个独立信息）。

        设计决策：规则只保留**多问号**这一个可靠信号（"…？…？"= 两个独立子问题），
        其余（"X和Y的区别"等）交给 LLM decompose_plan 判断（它会输出
        decomposed:false 若不需要拆）。这样避免规则误伤 easy 题（guardrail）。
        """
        # 多个问号 → 多个独立子问题（最可靠信号）
        n_q = question.count("？") + question.count("?")
        if n_q >= 2:
            parts = [p for p in question.replace("?", "？").split("？") if p.strip()]
            if len(parts) >= 2:
                ents = [set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", p.lower())) for p in parts]
                # 分句共享英文实体 → 同主题追问（"DICOM…？转换公式…？"）
                if ents[0] and ents[1] and (ents[0] & ents[1]):
                    return False
                # 后分句是极短裸问（≤8 字符，无新主题名词）→ 追问而非独立子问题
                # （"转换公式是什么？"=7；对比"急性血栓的征象有哪些？"=10 含主题名词）
                if len(parts[1]) <= 8 and not ents[1]:
                    return False
            return True
        # 单问号 + 明确对比词（"A和B的区别"）→ 两个不同实体
        if any(k in question for k in ("区别", "对比", "异同")):
            return True
        # 单问号 + "A和B分别" + 英文专名（"U-Net和TransUNet"）→ 不同实体
        if ("和" in question or "与" in question) and ("分别" in question or "各" in question):
            return bool(re.search(r"[A-Za-z][A-Za-z0-9\-]{2,}", question))
        return False

    @staticmethod
    def _entity_overlap(question: str, candidates: list[dict], top_n: int = 5) -> float:
        """关键实体覆盖率：问题中的区分性实体在候选文本中出现的比例（0-1）

        区分性实体 = 英文 token（≥3 字符）+ 中文 2-4 字片段（排除通用词）。
        覆盖率 = 命中的实体数 / 总实体数。用于 Policy Node 的客观信号。
        """
        cand_text = " ".join(c["text"][:800] for c in candidates[:top_n]).lower()
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
            return 0.0
        hit = sum(1 for e in entities if e in cand_text)
        return hit / len(entities)

    def run(self, question: str, fetch_k: int = 20, verbose: bool = True) -> dict[str, Any]:
        """执行 Agentic 检索循环（v2: hop-aware）

        v2 流程（Step 13）：
          1. 初始检索 → evidence_bank
          2. Policy：判断是否需要 plan（DECOMPOSE）或直接评估
          3. 有 plan → 逐 hop targeted retrieve → completeness check
          4. ACCEPT 条件：relevant AND complete AND supported（13C）
          5. RETRIEVE 携带 target_hop（13D）
          6. ABSTAIN = 预算耗尽（13F）

        Returns:
            {
                "state": AgentState,        # 完整状态（含 plan/hops/completeness）
                "answer": str,              # 生成回答 / 拒答
                "sources": list[dict],      # final_evidence
                "route": list[str],         # 决策路径
                "iterations": int,
                "abstained": bool,
            }
        """
        state = AgentState(original_query=question)
        state.retrieval_budget = self.max_iterations + 2  # 13F：预算
        t0 = time.time()

        # ── Iteration 0：初始检索 + 评分 ──
        state.route.append("RETRIEVE")
        initial = self.hybrid_search(question, fetch_k=fetch_k, note="initial")
        state.retrieval_history.append(
            {"query": question, "sources": initial, "iteration": state.iteration, "reason": "initial"}
        )
        self._accumulate_evidence(state, initial)
        state.iteration += 1
        state.retrieval_budget -= 1

        grade = self.evidence_grade(question, state.candidates)
        state.evidence_score = grade["evidence_score"]
        status, decision, pmode = self.policy(question, state, grade)
        state.evidence_status = status

        while decision in ("RETRIEVE", "DECOMPOSE") and state.retrieval_budget > 0:
            state.route.append(decision)
            if verbose:
                print(f"  🤖 [{state.iteration}] {decision}: {grade['reason'][:60]}")

            if decision == "DECOMPOSE":
                # ── 13E：structured plan → 逐 hop targeted retrieve ──
                state.decompose_attempted = True
                plan = self.decompose_plan(question)
                if verbose:
                    print(f"      🔗 plan: {plan}")
                if plan:
                    self._init_hops_from_plan(state, plan)
                    for hop in state.hops:
                        if state.retrieval_budget <= 0:
                            break
                        self._targeted_retrieve(state, hop, fetch_k=fetch_k)
                        state.iteration += 1
                        state.retrieval_budget -= 1
                    # plan 执行完 → completeness check
                    completeness, missing = self._compute_completeness(state)
                    state.completeness = completeness
                    if missing and state.retrieval_budget > 0:
                        # 还有缺失 hop → targeted 再捞
                        for hop in missing:
                            if state.retrieval_budget <= 0:
                                break
                            self._targeted_retrieve(state, hop, fetch_k=fetch_k)
                            state.iteration += 1
                            state.retrieval_budget -= 1
                        completeness, missing = self._compute_completeness(state)
                        state.completeness = completeness
                else:
                    # 拆解失败（LLM 认为不需要拆 / 输出异常）→ 不再重复 DECOMPOSE。
                    # 标记 plan 已尝试过，下次 policy 直接走单跳路径（防死循环）。
                    state.hops = []  # 清空 plan，policy 的 _is_multi_part and not state.hops 不再触发
                    state.plan = []
                    # 退化为 targeted 单跳检索
                    more = self.hybrid_search(question, fetch_k=fetch_k, note="retry-after-decompose-fail")
                    state.retrieval_history.append(
                        {"query": question, "sources": more, "iteration": state.iteration, "reason": "retry"}
                    )
                    self._accumulate_evidence(state, more)
                    state.iteration += 1
                    state.retrieval_budget -= 1
            else:  # RETRIEVE
                # ── 13D：targeted retrieval（携带缺失 hop 的 subquery）──
                target = grade.get("target_hop")
                if target and state.hops:
                    hop = next((h for h in state.hops if h.hop_id == target["hop_id"]), None)
                    if hop:
                        self._targeted_retrieve(state, hop, fetch_k=fetch_k)
                        state.iteration += 1
                        state.retrieval_budget -= 1
                    else:
                        more = self.hybrid_search(target.get("query", question), fetch_k=fetch_k, note="targeted")
                        state.retrieval_history.append(
                            {
                                "query": target.get("query", question),
                                "sources": more,
                                "iteration": state.iteration,
                                "reason": "targeted",
                            }
                        )
                        self._accumulate_evidence(state, more)
                        state.iteration += 1
                        state.retrieval_budget -= 1
                else:
                    # 无明确 target → 换角度再检索（v1 兼容）
                    new_query = self._build_retrieval_variant(state, question)
                    if verbose:
                        print(f"      🔎 再检索: {new_query[:60]}")
                    more = self.hybrid_search(new_query, fetch_k=fetch_k, note=f"retrieve: {new_query}")
                    state.retrieval_history.append(
                        {"query": new_query, "sources": more, "iteration": state.iteration, "reason": "retrieve"}
                    )
                    self._accumulate_evidence(state, more)
                    state.iteration += 1
                    state.retrieval_budget -= 1

            grade = self.evidence_grade(question, state.candidates)
            state.evidence_score = grade["evidence_score"]
            status, decision, pmode = self.policy(question, state, grade)
            state.evidence_status = status

        # ── 终局决策（13F：ABSTAIN = 预算耗尽；预算内 ACCEPT 优先）──
        if decision == "ACCEPT":
            state.decision = "ACCEPT"
            state.route.append("ACCEPT")
            if verbose:
                print(f"  ✅ ACCEPT (completeness={state.completeness:.2f} score={state.evidence_score:.2f})")
        elif decision == "ABSTAIN" or state.retrieval_budget <= 0:
            # 预算耗尽或明确 ABSTAIN → 最终判定
            if decision == "ACCEPT":
                pass
            # 终局：completeness 检查给最后一次机会（预算耗尽前）
            if state.hops and state.completeness < 1.0:
                state.decision = "ABSTAIN"
                state.route.append("ABSTAIN")
                state.abstain_reason = grade.get("reason", f"completeness={state.completeness:.2f}")
                if verbose:
                    print(f"  🚫 ABSTAIN: {state.abstain_reason[:60]}")
            elif decision == "ABSTAIN":
                state.decision = "ABSTAIN"
                state.route.append("ABSTAIN")
                state.abstain_reason = grade.get("reason", "证据不足")
                if verbose:
                    print(f"  🚫 ABSTAIN: {state.abstain_reason[:60]}")
            else:
                # RETRIEVE/DECOMPOSE 但预算耗尽 → ABSTAIN
                state.decision = "ABSTAIN"
                state.route.append("ABSTAIN")
                state.abstain_reason = grade.get("reason", "检索预算耗尽")
                if verbose:
                    print(f"  🚫 ABSTAIN（预算耗尽）: {state.abstain_reason[:60]}")
        else:
            # 预算还有但 policy 未定 → ABSTAIN（安全兜底）
            state.decision = "ABSTAIN"
            state.route.append("ABSTAIN")
            state.abstain_reason = grade.get("reason", "证据不足")
            if verbose:
                print(f"  🚫 ABSTAIN: {state.abstain_reason[:60]}")

        # ── 生成 / 拒答 ──
        if state.decision == "ACCEPT":
            state.final_evidence = self._select_final_evidence(question, state.candidates, fetch_k, state=state)
            answer = self.generate(question, state.final_evidence)
        else:
            state.final_evidence = state.candidates[:fetch_k]
            answer = self._abstain_response(question, state.abstain_reason)

        return {
            "state": state,
            "answer": answer,
            "sources": state.final_evidence,
            "route": state.route,
            "iterations": state.iteration,
            "abstained": state.decision == "ABSTAIN",
            "elapsed": round(time.time() - t0, 2),
        }

    def _is_topically_related(self, question: str, candidates: list[dict]) -> bool:
        """主题相关性检查：问题的独特实体是否出现在候选文本中

        比 2-gram 共享更严格——要求"区分性实体"命中：
          - 英文 token（≥3 字符，如 YEARS / sPESI / COVID）
          - 中文专有实体（滑动窗口 2-4 字，排除通用词）

        True = 主题相关（证据单薄应降级生成，不拒答）
        False = 主题无关（OOD，保持 ABSTAIN）
        """
        if _is_out_of_domain(question):
            return False
        cand_text = " ".join(c["text"][:600] for c in candidates[:5]).lower()

        # 英文区分性 token
        for tok in re.findall(r"[a-z][a-z0-9\-]{2,}", question.lower()):
            if tok in cand_text:
                return True

        # 中文区分性实体：滑动窗口 3-4 字（2 字太敏感——"心血""肺栓"等
        # 在医学文档中高频出现，会把 OOD 误判为相关）
        for n in (4, 3):
            for i in range(len(question) - n + 1):
                frag = question[i : i + n]
                if not re.fullmatch(r"[一-鿿]+", frag):
                    continue
                if frag in _GENERIC_TERMS:
                    continue
                if frag in cand_text:
                    return True
        return False

    def _select_final_evidence(
        self, question: str, candidates: list[dict], k: int, state: AgentState | None = None
    ) -> list[dict]:
        """从累积候选中选择最终证据

        v1 用 cross-encoder reranker（若有）对候选重排取 top-k；
        v2（Step 13）有 plan 时从 evidence_by_hop 合并（每 hop 取 top evidence），
        保证 hop 级命中的证据不被原始问题的 rerank 挤掉（bh_partial_01 修复）。
        """
        # v2: 有 plan → 从 evidence_by_hop 合并
        if state is not None and state.hops:
            merged: list[dict] = []
            seen = set()
            for hop in state.hops:
                for c in state.evidence_by_hop.get(hop.hop_id, []):
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        merged.append(c)
            if merged:
                return merged[:k]
        reranker = self.reranker or getattr(self.retriever, "_reranker", None)
        if reranker is not None and getattr(reranker, "model_ready", False):
            try:
                return reranker.rerank(question, list(candidates), k)
            except Exception:
                pass
        return candidates[:k]

    # ══════════════════════════════════════════════════
    #  辅助
    # ══════════════════════════════════════════════════

    def _build_retrieval_variant(self, state: AgentState, question: str, deeper: bool = False) -> str:
        """构造检索变体：从候选 chunk 提取高频实体补充查询

        第一版不做 LLM Rewrite（已冻结），用轻量规则：
          取候选 top5 文本中出现的高频 2-4 字片段加入查询。
        """
        from collections import Counter

        freq: Counter = Counter()
        for c in state.candidates[:10]:
            text = c["text"]
            # 提取 2-4 字中文词 + 英文术语
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

    @staticmethod
    def _abstain_response(question: str, reason: str = "") -> str:
        """拒答：说明证据不足"""
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

    # ── State 序列化 ──

    def state_to_dict(self, state: AgentState) -> dict[str, Any]:
        """把 AgentState 转可 JSON 序列化 dict"""
        return {
            "original_query": state.original_query,
            "retrieval_history": [
                {
                    "query": h["query"],
                    "iteration": h["iteration"],
                    "reason": h["reason"],
                    "num_sources": len(h["sources"]),
                }
                for h in state.retrieval_history
            ],
            "candidates": [
                {"id": c["id"], "filename": c["metadata"].get("filename", ""), "text": c["text"][:200]}
                for c in state.candidates
            ],
            "evidence_score": state.evidence_score,
            "evidence_status": state.evidence_status,
            "completeness": state.completeness,
            "plan": state.plan,
            "hops": [
                {
                    "hop_id": h.hop_id,
                    "subquery": h.subquery,
                    "support_status": h.support_status,
                    "evidence_ids": h.evidence_ids,
                    "evidence_score": round(h.evidence_score, 3),
                    "retrieval_attempts": h.retrieval_attempts,
                }
                for h in state.hops
            ],
            "evidence_by_hop": {str(k): [c["id"] for c in v] for k, v in state.evidence_by_hop.items()},
            "route": state.route,
            "iteration": state.iteration,
            "final_evidence": [c["id"] for c in state.final_evidence],
            "decision": state.decision,
            "abstain_reason": state.abstain_reason,
        }

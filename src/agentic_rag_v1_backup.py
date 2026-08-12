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
class AgentState:
    """Agentic RAG 状态"""

    original_query: str = ""
    retrieval_history: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    evidence_score: float = 0.0
    route: list[str] = field(default_factory=list)
    iteration: int = 0
    final_evidence: list[dict] = field(default_factory=list)
    decision: str = ""  # ACCEPT / RETRIEVE / DECOMPOSE / ABSTAIN
    abstain_reason: str = ""


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
        # LLM 不可用时，先做领域判定：明显非医学问题直接 insufficient
        if _is_out_of_domain(question):
            return {
                "decision": "insufficient",
                "reason": "问题明显超出医学知识库领域（规则判定）",
                "evidence_score": 0.0,
                "mode": "rule",
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
        """工具 4：生成回答（基于最终证据）"""
        from .generator import build_rag_prompt

        if not chunks:
            return self._abstain_response(question)
        messages, source_map, relevance = build_rag_prompt(question, chunks)
        gen = self.generator.generate_structured((messages, source_map, relevance), self_reflect=False)
        return gen["raw"]

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

    def _decide(self, grade: dict) -> str:
        """把 grader 输出映射为 Agent 决策"""
        if grade["decision"] == "sufficient":
            return "ACCEPT"
        if grade["decision"] == "needs_decomposition":
            return "DECOMPOSE"
        return "RETRIEVE"

    def run(self, question: str, fetch_k: int = 20, verbose: bool = True) -> dict[str, Any]:
        """执行 Agentic 检索循环

        Returns:
            {
                "state": AgentState,        # 完整状态（可序列化）
                "answer": str,              # 生成回答 / 拒答
                "sources": list[dict],      # final_evidence
                "route": list[str],         # 决策路径
                "iterations": int,
                "abstained": bool,
            }
        """
        state = AgentState(original_query=question)
        t0 = time.time()

        # ── Iteration 0：初始检索 + 评分 ──
        state.route.append("RETRIEVE")
        initial = self.hybrid_search(question, fetch_k=fetch_k, note="initial")
        state.retrieval_history.append(
            {"query": question, "sources": initial, "iteration": state.iteration, "reason": "initial"}
        )
        self._dedup_accumulate(state, initial)
        state.iteration += 1

        grade = self.evidence_grade(question, state.candidates)
        state.evidence_score = grade["evidence_score"]
        decision = self._decide(grade)

        while decision in ("RETRIEVE", "DECOMPOSE") and state.iteration < self.max_iterations:
            if verbose:
                print(f"  🤖 [{state.iteration}] {decision}: {grade['reason'][:60]}")

            if decision == "DECOMPOSE":
                # ── 拆解子问题后分别检索 ──
                sub_questions = self.decompose(question)
                if verbose:
                    print(f"      🔗 拆解: {sub_questions}")
                if sub_questions:
                    for sub in sub_questions:
                        sub_results = self.hybrid_search(sub, fetch_k=fetch_k, note=f"decompose: {sub}")
                        state.retrieval_history.append(
                            {"query": sub, "sources": sub_results, "iteration": state.iteration, "reason": "decompose"}
                        )
                        self._dedup_accumulate(state, sub_results)
                else:
                    # 拆解失败 → 退化为再检索
                    more = self.hybrid_search(question, fetch_k=fetch_k, note="retry-after-decompose-fail")
                    state.retrieval_history.append(
                        {"query": question, "sources": more, "iteration": state.iteration, "reason": "retry"}
                    )
                    self._dedup_accumulate(state, more)
            else:  # RETRIEVE
                # ── 换角度再检索：用候选中的高频实体/术语补一个新查询 ──
                new_query = self._build_retrieval_variant(state, question)
                if verbose:
                    print(f"      🔎 再检索: {new_query[:60]}")
                more = self.hybrid_search(new_query, fetch_k=fetch_k, note=f"retrieve: {new_query}")
                state.retrieval_history.append(
                    {"query": new_query, "sources": more, "iteration": state.iteration, "reason": "retrieve"}
                )
                self._dedup_accumulate(state, more)

            state.iteration += 1
            grade = self.evidence_grade(question, state.candidates)
            state.evidence_score = grade["evidence_score"]
            decision = self._decide(grade)

        # ── 终局决策 ──
        if decision == "ACCEPT":
            state.decision = "ACCEPT"
            state.route.append("ACCEPT")
            if verbose:
                print(f"  ✅ ACCEPT (score={state.evidence_score:.2f})")
        else:
            # 证据仍不足 → 最后再试一次检索（若还有迭代额度）或 Abstain
            if state.iteration < self.max_iterations:
                new_query = self._build_retrieval_variant(state, question, deeper=True)
                more = self.hybrid_search(new_query, fetch_k=fetch_k * 2, note="final-retry")
                state.retrieval_history.append(
                    {"query": new_query, "sources": more, "iteration": state.iteration, "reason": "final-retry"}
                )
                self._dedup_accumulate(state, more)
                state.iteration += 1
                grade = self.evidence_grade(question, state.candidates)
                state.evidence_score = grade["evidence_score"]
                if grade["decision"] == "sufficient":
                    state.decision = "ACCEPT"
                    state.route.append("ACCEPT")
                    if verbose:
                        print(f"  ✅ ACCEPT（最终检索补救，score={state.evidence_score:.2f})")
                else:
                    state.decision = "ABSTAIN"
                    state.route.append("ABSTAIN")
                    state.abstain_reason = grade["reason"]
                    if verbose:
                        print(f"  🚫 ABSTAIN: {grade['reason'][:60]}")
            else:
                state.decision = "ABSTAIN"
                state.route.append("ABSTAIN")
                state.abstain_reason = grade["reason"]
                if verbose:
                    print(f"  🚫 ABSTAIN: {grade['reason'][:60]}")

        # ── 终局 re-grade（扩大候选视野后给最后一次机会）──
        if state.decision == "ABSTAIN":
            # 主题相关性检查：候选与问题共享词面 → 主题相关，证据单薄不等于无关
            if self._is_topically_related(question, state.candidates):
                grade_final = self.evidence_grade(question, state.candidates[:12])
                if grade_final["decision"] in ("sufficient", "needs_decomposition"):
                    state.decision = "ACCEPT"
                    state.route.append("ACCEPT")
                    state.evidence_score = grade_final["evidence_score"]
                    if verbose:
                        print(f"  ✅ ACCEPT（终局 re-grade，score={grade_final['evidence_score']:.2f})")
                else:
                    # 主题相关但证据单薄 → 降级 ACCEPT（宁可生成 + uncertainty，不可错杀）
                    state.decision = "ACCEPT"
                    state.route.append("ACCEPT")
                    state.evidence_score = grade_final["evidence_score"]
                    if verbose:
                        print(f"  ⚠️ ACCEPT（主题相关但证据单薄，score={grade_final['evidence_score']:.2f})")
            else:
                # 主题无关 → 保持 ABSTAIN
                if verbose:
                    print(f"  🚫 ABSTAIN（主题无关）: {state.abstain_reason[:60]}")

        # ── 生成 / 拒答 ──
        if state.decision == "ACCEPT":
            state.final_evidence = self._select_final_evidence(question, state.candidates, fetch_k)
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

    def _select_final_evidence(self, question: str, candidates: list[dict], k: int) -> list[dict]:
        """从累积候选中选择最终证据

        v1 用 cross-encoder reranker（若有）对候选重排取 top-k；
        无 reranker 时退化为候选前 k 个。
        """
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
            "route": state.route,
            "iteration": state.iteration,
            "final_evidence": [c["id"] for c in state.final_evidence],
            "decision": state.decision,
            "abstain_reason": state.abstain_reason,
        }

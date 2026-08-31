"""
Step 15: End-to-End Grounded Answer Evaluation（claim 级）

研究问题：RAG lifecycle 的最后一段——最终生成答案中的 factual claims
是否由 final_evidence 支撑？检索命中 ≠ 答案正确，答案正确 ≠ 每句都 grounded。

指标（6 项 + Correct Abstention）：
  1. Answer Correctness       最终答案是否包含预期关键数值/结论（宽松匹配）
  2. Groundedness             答案 claims 中被 evidence 支撑的比例（LLM-Judge）
  3. Evidence/Citation Support 引用编号是否有效 + 每 claim 的引用是否真实支撑
  4. Completeness             证据对问题覆盖的完整度（hop gold 命中比例）
  5. Unsupported Claim Rate   答案中无证据支撑的 claim 比例（显式记录，不隐藏）
  6. Correct Abstention       OOD 正确拒答（与 Step 12/13/14 同口径）

裁判：
  - claim 抽取 + grounded 判定 = LLM-as-Judge（deepseek-chat，temperature=0）
  - LLM 不可用时降级规则判定（claim 级 char-overlap 交叉验证）

设计原则（与 Step 15 目标一致）：
  - 不把 retrieval hit 当成最终质量——评测最终答案是否 grounded
  - 逐 claim 判定并显式记录每一条 unsupported claim（便于 Failure Anatomy）
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .rescue_metrics import evidence_recall_at_k, hop_gold_ids

# ══════════════════════════════════════════════════
#  LLM-as-Judge Prompt（claim 抽取 + grounded 判定）
# ══════════════════════════════════════════════════

CLAIM_EVAL_SYSTEM_PROMPT = """\
你是一个严格的事实核查员。给定用户问题、AI 回答和参考证据片段，逐条判断回答中的事实性陈述（claim）是否由证据支撑。

## 判定规则
- 把回答按"事实性陈述"切分为若干 claim（数字、定义、机制、步骤、结论等）
- 对每个 claim 判定支撑状态：
  - supported:    证据中存在直接支撑该 claim 的内容
  - unsupported:  证据中没有支撑，或与证据矛盾（虚构/幻觉）
  - unverifiable: 陈述无实质事实内容（如"综上所述""以上是…"等衔接语），不计入 grounded 分母
- 注意：回答可能引用了 [N] 编号，请以**证据文本内容**为准判断支撑，而非只看编号

## 输出（严格 JSON，不要其他内容）
{"claims": [{"text": "claim 原文", "status": "supported|unsupported|unverifiable"}], "summary": {"supported": 3, "unsupported": 1, "unverifiable": 0}}
"""


def _claims_user_prompt(question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    refs = []
    for i, s in enumerate(sources, 1):
        refs.append(f"[{i}] {s.get('text', '')[:400]}")
    ref_text = "\n\n".join(refs) if refs else "（无参考证据）"
    return f"""## 用户问题
{question}

## AI 回答
{answer}

## 参考证据片段
{ref_text}

请逐条判定回答中的事实性陈述是否由证据支撑："""


@dataclass
class ClaimResult:
    """单个 claim 的判定结果"""

    text: str
    status: str  # supported / unsupported / unverifiable
    evidence_idx: list[int] = field(default_factory=list)


@dataclass
class GroundedCase:
    """单题的 grounded 评测结果（可 JSON 序列化）"""

    id: str = ""
    type: str = ""
    question: str = ""
    answer: str = ""
    abstained: bool = False
    answer_correct: bool = False
    claims: list[ClaimResult] = field(default_factory=list)
    evidence_recall: float = 0.0
    completeness: float = 0.0
    citation_valid: bool = True
    mode: str = "llm"  # llm / rule
    judge_raw: str = ""  # judge 原始输出留档（P1-6 分歧复盘）

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "answer": self.answer[:500],
            "abstained": self.abstained,
            "answer_correct": self.answer_correct,
            "claims": [{"text": c.text, "status": c.status, "evidence_idx": c.evidence_idx} for c in self.claims],
            "evidence_recall": self.evidence_recall,
            "completeness": self.completeness,
            "citation_valid": self.citation_valid,
            "mode": self.mode,
            "judge_raw": self.judge_raw[:2000],
        }

    # ── 派生指标 ──
    @property
    def groundedness(self) -> float:
        """有实质内容的 claim 中被支撑的比例（0-1）；无 claim → 0"""
        factual = [c for c in self.claims if c.status != "unverifiable"]
        if not factual:
            return 0.0
        return sum(1 for c in factual if c.status == "supported") / len(factual)

    @property
    def unsupported_claims(self) -> list[ClaimResult]:
        return [c for c in self.claims if c.status == "unsupported"]

    @property
    def unsupported_claim_rate(self) -> float:
        factual = [c for c in self.claims if c.status != "unverifiable"]
        if not factual:
            return 0.0
        return sum(1 for c in factual if c.status == "unsupported") / len(factual)


# ══════════════════════════════════════════════════
#  Claim 级判定
# ══════════════════════════════════════════════════


def _parse_claims_response(response: str) -> list[ClaimResult] | None:
    """解析 LLM claim 判定 JSON；失败返回 None（走规则降级）"""
    m = re.search(r"\{[\s\S]*\}", response)
    if not m:
        return None
    try:
        data = json.loads(m.group())
        claims = []
        for c in data.get("claims", []):
            text = str(c.get("text", "")).strip()
            status = str(c.get("status", "")).strip()
            if not text:
                continue
            if status not in ("supported", "unsupported", "unverifiable"):
                continue
            claims.append(ClaimResult(text=text, status=status))
        if claims:
            return claims
    except Exception:
        pass
    return None


def judge_claims(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    generator=None,
) -> tuple[list[ClaimResult], str, str]:
    """LLM-as-Judge 判定回答中的 claims 是否被证据支撑

    Returns:
        (claims, mode, judge_raw)  mode ∈ {"llm", "rule"}
        judge_raw 为 judge 原始输出（截断留档，供 rules vs LLM 分歧复盘，
        P1-6）；规则降级时为空。
    """
    if generator is not None and answer and not answer.startswith("[OPERATIONAL_ERROR]"):
        try:
            raw = generator.chat(
                messages=[
                    {"role": "system", "content": CLAIM_EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": _claims_user_prompt(question, answer, sources)},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            claims = _parse_claims_response(raw)
            if claims is not None:
                return claims, "llm", raw
        except Exception:
            pass
    return _rule_based_claims(question, answer, sources), "rule", ""


def _rule_based_claims(question: str, answer: str, sources: list[dict[str, Any]]) -> list[ClaimResult]:
    """规则降级：claim 级 char-overlap 交叉验证（eval/judge.py 的 N-gram 思想）

    把回答切成句子（事实性陈述），每句与证据文本做字符重叠匹配；
    ≥60% 重叠 → supported，否则 unsupported（无证据或拒答回答 → 全部 unverifiable）。
    """
    if not answer or "知识库中未找到" in answer or answer.startswith("[OPERATIONAL_ERROR]"):
        return []
    doc_text = " ".join(s.get("text", "") for s in sources)
    claims: list[ClaimResult] = []
    sentences = [s.strip() for s in re.split(r"[。！？；\n]+", answer) if len(s.strip()) >= 6]
    for sent in sentences:
        # 非事实性片段（markdown 标题/纯引用行/衔接语）
        if re.match(r"^[*>\-#\d.\[\]（）()\s]+$", sent) or "**" in sent[:4]:
            claims.append(ClaimResult(text=sent, status="unverifiable"))
            continue
        chars = set(sent)
        if not chars:
            claims.append(ClaimResult(text=sent, status="unverifiable"))
            continue
        overlap = len(chars & set(doc_text)) / len(chars)
        status = "supported" if overlap >= 0.6 else "unsupported"
        claims.append(ClaimResult(text=sent, status=status))
    return claims


def _citation_valid(answer: str, sources: list[dict[str, Any]]) -> bool:
    """引用有效性：答案引用的 [N] 都在证据编号范围内"""
    from src.generator import validate_citations

    source_map = {str(i + 1): {} for i in range(len(sources))}
    if not source_map:
        return not re.search(r"\[\d+\]", answer)
    try:
        cv = validate_citations(answer, source_map)
        return not cv.get("has_invalid_citations", False)
    except Exception:
        return True


# ══════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════


def compute_grounded_metrics(
    cases: list[dict[str, Any]],
    generator=None,
    k: int = 5,
) -> dict[str, Any]:
    """对逐题 case 计算 End-to-End Grounded Answer 指标

    每题 case 需含:
        {
          "question": dict,        # benchmark 条目（id/type/hops/final_answer）
          "answer": str,           # Agent 最终回答
          "sources": list[dict],   # final_evidence（TopK）
          "abstained": bool,
        }

    Returns: 6 项核心指标 + Correct Abstention + 逐题 claim 明细。
    """
    n = len(cases)
    agg = {
        "n": n,
        "answer_correct": 0,
        "groundedness": 0.0,
        "evidence_recall": 0.0,
        "completeness": 0.0,
        "unsupported_claim_total": 0,
        "claim_total": 0,
        "unsupported_claim_rate": 0.0,
        "citation_valid": 0,
        "ood_total": 0,
        "ood_reject_ok": 0,
        "answerable_total": 0,
        "false_abstain": 0,
    }
    details: list[GroundedCase] = []
    all_unsupported: list[dict] = []

    for c in cases:
        q = c["question"]
        qtype = q.get("type", "?")
        answer = c.get("answer", "")
        sources = c.get("sources", [])[:k]
        abstained = c.get("abstained", False)
        expected = q.get("final_answer", "")

        case = GroundedCase(
            id=q.get("id", ""),
            type=qtype,
            question=q.get("question", ""),
            answer=answer,
            abstained=abstained,
        )

        # 1. Answer Correctness（宽松匹配，与 Step 12 同口径）
        expected_s = str(expected).strip()
        case.answer_correct = bool(expected_s) and (expected_s in answer)

        # 2. Claim 级 grounded 判定（LLM → 规则降级）
        if not abstained and answer and not answer.startswith("[OPERATIONAL_ERROR]"):
            case.claims, case.mode, case.judge_raw = judge_claims(q.get("question", ""), answer, sources, generator)

        # 3. Evidence / Citation Support
        all_gold = set()
        for hg in hop_gold_ids(q):
            all_gold |= hg
        case.evidence_recall = evidence_recall_at_k(sources, all_gold, k)
        case.completeness = case.evidence_recall if all_gold else 0.0
        case.citation_valid = _citation_valid(answer, sources)

        # ── 汇总 ──
        agg["answer_correct"] += case.answer_correct
        agg["groundedness"] += case.groundedness
        agg["evidence_recall"] += case.evidence_recall
        agg["completeness"] += case.completeness
        agg["citation_valid"] += case.citation_valid

        factual = [cl for cl in case.claims if cl.status != "unverifiable"]
        agg["claim_total"] += len(factual)
        uns = len(case.unsupported_claims)
        agg["unsupported_claim_total"] += uns
        for cl in case.unsupported_claims:
            all_unsupported.append({"id": case.id, "question": case.question, "claim": cl.text, "answer": answer[:200]})

        # 4. Correct Abstention / False Abstain（与 Step 12 同口径）
        if qtype == "unsupported_ood":
            agg["ood_total"] += 1
            agg["ood_reject_ok"] += abstained
        else:
            agg["answerable_total"] += 1
            if abstained:
                agg["false_abstain"] += 1

        details.append(case)

    n_ans = max(agg["answerable_total"], 1)
    n_ood = max(agg["ood_total"], 1)
    total_factual = max(agg["claim_total"], 1)
    agg["unsupported_claim_rate"] = round(agg["unsupported_claim_total"] / total_factual, 3)

    return {
        "n": n,
        "Answer Correctness": f"{agg['answer_correct']}/{n}",
        "Groundedness": round(agg["groundedness"] / n, 3),
        "Evidence/Citation Support": round(agg["evidence_recall"] / n, 3),
        "Completeness": round(agg["completeness"] / n, 3),
        "Unsupported Claim Rate": agg["unsupported_claim_rate"],
        "Correct Abstention": f"{agg['ood_reject_ok']}/{agg['ood_total']}",
        "False Abstain": f"{agg['false_abstain']}/{agg['answerable_total']}",
        "Citation Valid Rate": round(agg["citation_valid"] / n, 3),
        "claim_total": agg["claim_total"],
        "unsupported_claim_total": agg["unsupported_claim_total"],
        "unsupported_claims": all_unsupported,
        "details": [d.to_dict() for d in details],
    }

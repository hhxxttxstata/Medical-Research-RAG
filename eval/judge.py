"""
LLM-as-Judge 回答质量评估

对 RAG 系统的回答从多个维度评分：
  - faithfulness:    回答是否忠实于参考文档（不虚构、不歪曲）
  - completeness:    回答是否完整覆盖了问题的各个方面
  - helpfulness:     回答对用户是否有实际帮助
  - citation_accuracy: 引用标注是否准确（假的引用编号）

使用 create_generator() 获取 LLM（与系统主流程一致），无 API Key 时自动 fallback 到
基于 chunk 内容交叉验证的规则评估。

Changes from previous version:
  - 改用 create_generator() 而非硬编码 _call_openai，适配 DeepSeek/Ollama
  - 维度权重调整：faithfulness=0.30, completeness=0.30, helpfulness=0.20, citation_accuracy=0.20
  - 输出增加每个维度的 reasoning 文本
  - rule-based 改为基于 chunk 文本的 N-gram 交叉验证
"""

import re
from typing import Any

from src.generator import LLMGenerator, create_generator

# ── 评分维度 ──────────────────────────────────

DIMENSIONS = {
    "faithfulness": {
        "name": "忠实度",
        "description": "回答是否忠实于参考文档，不虚构、不歪曲事实",
        "weight": 0.30,
    },
    "completeness": {
        "name": "完整性",
        "description": "回答是否完整覆盖了问题的各个方面",
        "weight": 0.30,
    },
    "helpfulness": {
        "name": "有用性",
        "description": "回答对用户是否有实际帮助，是否清晰易懂",
        "weight": 0.20,
    },
    "citation_accuracy": {
        "name": "引用准确性",
        "description": "引用标注是否准确，不引用不存在的来源编号",
        "weight": 0.20,
    },
}


def judge_default_score() -> dict[str, float]:
    """无有效评估时返回默认分（用于占位）"""
    return {
        "faithfulness": 0.0,
        "completeness": 0.0,
        "helpfulness": 0.0,
        "citation_accuracy": 0.0,
        "overall": 0.0,
        "mode": "none",
    }


# ── 基于 chunk 内容的规则评分（fallback） ──


def _extract_key_facts(text: str, max_ngram: int = 3) -> set:
    """从文本中提取关键 N-gram 作为事实片段

    用于交叉验证回答是否在参考文档中有依据。
    """
    # 按中文句号/英文句点/问号/感叹号/分号拆分句子
    sentences = re.split(r"[。！？；\n]+", text)
    facts = set()
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue
        # 提取长度在 12-80 字之间的关键短语
        for n in range(max_ngram, 1, -1):
            # 按标点分割成更小的片段
            fragments = re.split(r"[，,、：:；;](?![^)]*\))", sent)
            for frag in fragments:
                frag = frag.strip()
                # 过滤太短或太长的
                if 12 <= len(frag) <= 80:
                    facts.add(frag)
                # 也加入完整句子中较长的部分
                if 20 <= len(sent) <= 120:
                    facts.add(sent)
    return facts


def _rule_based_judge(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    relevance_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """基于 chunk 内容交叉验证的规则评分

    替代原有关键词匹配，使用 N-gram 重叠率评估忠实度。
    """
    # ── 提取参考文档中的所有关键事实 ──
    doc_facts = set()
    for s in sources:
        doc_facts.update(_extract_key_facts(s.get("text", "")))

    # ── 提取回答中的关键事实 ──
    answer_facts = _extract_key_facts(answer)

    # ── faithfulness: 回答中的事实有多少能在文档中找到依据 ──
    if answer_facts and doc_facts:
        matched = 0
        for af in answer_facts:
            # 检查该事实是否在任何一个文档片段中出现
            for df in doc_facts:
                # 使用字符重叠率（中文场景比 word overlap 更稳定）
                overlap_chars = len(set(af) & set(df))
                shorter = min(len(af), len(df))
                if shorter > 0 and overlap_chars / shorter >= 0.6:
                    matched += 1
                    break
        match_ratio = matched / len(answer_facts)
        faithfulness = round(1.0 + match_ratio * 4.0, 1)  # 映射到 1-5
    else:
        faithfulness = 2.5  # 无文档或无回答，保守分

    # ── completeness: 基于回答是否覆盖了问题的关键词 ──
    question_keywords = set(w for w in re.split(r"[？?，,。.、\s]+", question) if len(w) >= 2)
    answer_keywords = set(w for w in re.split(r"[？?，,。.、\s\n]+", answer) if len(w) >= 2)
    if question_keywords:
        keyword_coverage = len(question_keywords & answer_keywords) / len(question_keywords)
        completeness = round(1.0 + keyword_coverage * 4.0, 1)
    else:
        completeness = 3.0

    # 回答长度也影响完整性
    answer_length = len(answer)
    if answer_length > 300:
        completeness = min(5.0, completeness + 0.5)
    elif answer_length < 50:
        completeness = max(1.0, completeness - 0.5)

    # ── helpfulness: 基于是否结构化 + 有无建议 ──
    helpfulness = 3.0
    if answer_length > 100:
        helpfulness += 0.5
    if "**结论：**" in answer or "**建议**" in answer or "建议" in answer:
        helpfulness += 0.5
    if "**依据：**" in answer or "**引用来源：**" in answer:
        helpfulness += 0.5
    helpfulness = min(5.0, helpfulness)

    # ── citation_accuracy: 基于引用验证 ──
    source_map = {}
    for i, s in enumerate(sources):
        source_map[str(i + 1)] = {
            "filename": s.get("metadata", {}).get("filename", ""),
            "score": s.get("score", 0),
        }

    from src.generator import validate_citations

    cv = validate_citations(answer, source_map) if source_map else {}

    if cv.get("has_invalid_citations"):
        citation_accuracy = 2.0
    elif cv.get("cited_valid"):
        citation_accuracy = 4.5
    else:
        citation_accuracy = 3.0

    # 综合分
    overall = (
        faithfulness * DIMENSIONS["faithfulness"]["weight"]
        + completeness * DIMENSIONS["completeness"]["weight"]
        + helpfulness * DIMENSIONS["helpfulness"]["weight"]
        + citation_accuracy * DIMENSIONS["citation_accuracy"]["weight"]
    )

    return {
        "faithfulness": round(faithfulness, 1),
        "completeness": round(completeness, 1),
        "helpfulness": round(helpfulness, 1),
        "citation_accuracy": round(citation_accuracy, 1),
        "overall": round(overall, 2),
        "faithfulness_reason": f"回答中 {matched if answer_facts else 0}/{len(answer_facts)} 个关键事实可在文档中找到依据"
        if answer_facts
        else "回答无关键事实可验证",
        "completeness_reason": f"问题关键词覆盖 {keyword_coverage:.0%}，回答长度 {answer_length} 字"
        if question_keywords
        else "",
        "helpfulness_reason": f"回答{'有' if '建议' in answer else '无'}建议，{'结构化' if '**' in answer else '未结构化'}",
        "citation_reason": f"引用验证: {'有效' if cv.get('cited_valid') else '无引用'}, {'无效' if cv.get('has_invalid_citations') else '无不准确引用'}",
        "mode": "rule",
    }


# ── LLM-as-Judge ──────────────────────────────


def _llm_judge_prompt(question: str, answer: str, sources: list[dict[str, Any]]) -> str:
    """构建 LLM Judge 的 prompt"""
    refs = []
    for i, s in enumerate(sources, 1):
        refs.append(f"[{i}] {s.get('text', '')[:300]}")
    ref_text = "\n\n".join(refs) if refs else "（无参考文档）"

    return f"""你是一个专业的 AI 回答质量评估员。请对以下 RAG 系统的回答进行评分。

## 用户问题
{question}

## 参考文档（检索结果片段）
{ref_text}

## AI 回答
{answer}

## 评分要求
请从以下 4 个维度对回答进行评分（1-5 分，可保留 1 位小数）：

1. **faithfulness（忠实度）**: 回答是否忠实于参考文档？
   - 5分：完全基于参考文档，无虚构
   - 3分：大部分正确但有少量无关补充
   - 1分：严重偏离或虚构内容

2. **completeness（完整性）**: 回答是否完整覆盖问题？
   - 5分：全面覆盖所有方面
   - 3分：覆盖了主要方面但不够深入
   - 1分：只覆盖了很少的内容

3. **helpfulness（有用性）**: 回答对用户是否有帮助？
   - 5分：清晰、结构化的回答，对用户非常有帮助
   - 3分：有一定帮助但不够清晰
   - 1分：几乎没有帮助

4. **citation_accuracy（引用准确性）**: 引用标注是否准确？
   - 5分：所有引用[编号]都在参考文档范围内
   - 3分：有少量不准确的引用
   - 1分：大量虚构引用

## 输出格式（必须严格按照以下 JSON 格式输出，不要包含其他内容）
{{"faithfulness": 分数, "completeness": 分数, "helpfulness": 分数, "citation_accuracy": 分数, "faithfulness_reason": "简要说明", "completeness_reason": "简要说明", "helpfulness_reason": "简要说明", "citation_reason": "简要说明"}}
"""


def _parse_judge_response(response: str) -> dict[str, float] | None:
    """解析 LLM Judge 的 JSON 响应"""
    json_match = re.search(r"\{[^}]+\}", response)
    if not json_match:
        return None
    try:
        import json

        scores = json.loads(json_match.group())
        required = ["faithfulness", "completeness", "helpfulness", "citation_accuracy"]
        if not all(k in scores for k in required):
            return None

        # 类型转换
        for k in required + ["faithfulness_reason", "completeness_reason", "helpfulness_reason", "citation_reason"]:
            if k in scores:
                if k.endswith("_reason"):
                    scores[k] = str(scores[k])
                else:
                    scores[k] = round(float(scores[k]), 1)

        scores["mode"] = "llm"
        scores["overall"] = round(
            scores["faithfulness"] * DIMENSIONS["faithfulness"]["weight"]
            + scores["completeness"] * DIMENSIONS["completeness"]["weight"]
            + scores["helpfulness"] * DIMENSIONS["helpfulness"]["weight"]
            + scores["citation_accuracy"] * DIMENSIONS["citation_accuracy"]["weight"],
            2,
        )
        return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def judge_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    generator: LLMGenerator | None = None,
    relevance_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对 RAG 回答进行质量评估

    Args:
        question: 用户问题
        answer: RAG 系统的回答
        sources: 检索到的文档片段列表
        generator: LLMGenerator 实例（可选，有 API Key 时启用 LLM Judge）
        relevance_info: 相关性判断信息（可选，用于规则评分）

    Returns:
        {
            "faithfulness": float,           # 0-5
            "completeness": float,           # 0-5
            "helpfulness": float,            # 0-5
            "citation_accuracy": float,      # 0-5
            "overall": float,                # 0-5 加权综合
            "faithfulness_reason": str,      # 评分理由
            "completeness_reason": str,
            "helpfulness_reason": str,
            "citation_reason": str,
            "mode": "llm" | "rule",          # 使用的评估方式
        }
    """
    # ── 尝试 LLM Judge ──
    # 优先使用传入的 generator，如果没有则尝试创建
    effective_generator = generator
    if effective_generator is None:
        try:
            effective_generator = create_generator()
        except Exception:
            pass

    if effective_generator is not None:
        try:
            prompt = _llm_judge_prompt(question, answer, sources)
            raw_response = effective_generator.chat(
                messages=[
                    {"role": "system", "content": "你是一个专业的回答质量评估员。请严格按照要求输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            scores = _parse_judge_response(raw_response)
            if scores is not None:
                return scores
        except Exception:
            pass

    # ── 规则 Judge（兜底） ──
    return _rule_based_judge(question, answer, sources, relevance_info)


def format_judge_scores(scores: dict[str, float]) -> str:
    """格式化评分为可读字符串"""
    if not scores or scores.get("mode") == "none":
        return "⚠️ 无法评估"

    lines = [
        f"📊 LLM-as-Judge 评分（模式: {scores.get('mode', '?')}）",
        f"   - 忠实度:      {scores.get('faithfulness', 0):.1f}/5 — {scores.get('faithfulness_reason', '')}",
        f"   - 完整性:      {scores.get('completeness', 0):.1f}/5 — {scores.get('completeness_reason', '')}",
        f"   - 有用性:      {scores.get('helpfulness', 0):.1f}/5 — {scores.get('helpfulness_reason', '')}",
        f"   - 引用准确度:  {scores.get('citation_accuracy', 0):.1f}/5 — {scores.get('citation_reason', '')}",
        f"   - 综合分:      {scores.get('overall', 0):.1f}/5",
    ]
    return "\n".join(lines)

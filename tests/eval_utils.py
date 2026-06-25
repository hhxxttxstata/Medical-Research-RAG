"""
Golden 数据集加载与评估辅助函数

用于 pytest 集成测试：
  1. load_golden_questions() — 从 JSON 加载测试题
  2. evaluate_retrieval()    — 单题检索评估
  3. evaluate_generation()   — 单题回答评估
"""

import json
import os
from typing import Any


def load_golden_questions(category: str | None = None) -> list[dict[str, Any]]:
    """从 tests/test_questions.json 加载 Golden 测试集

    Args:
        category: 筛选条件，None = 全部
            "exact_match" | "cross_doc" | "out_of_knowledge"

    Returns:
        符合条件的题目列表
    """
    path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    if category:
        questions = [q for q in questions if q["category"] == category]
    return questions


def questions_by_difficulty(questions: list[dict[str, Any]], difficulty: str) -> list[dict[str, Any]]:
    """按难度筛选"""
    return [q for q in questions if q.get("difficulty") == difficulty]


def evaluate_retrieval(pipeline, question: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """对单条 Golden 问题执行检索评估

    Args:
        pipeline: RAGPipeline 实例
        question: Golden 题目 dict
        top_k: 检索数量

    Returns:
        {
            "question_id": str,
            "hit_expected_doc": bool,      # 是否命中预期文档
            "num_retrieved": int,           # 实际返回数量
            "top_score": float,
            "avg_score": float,
            "scores": [float, ...],         # 所有检索分数
            "has_relevant": bool,           # 相关性判断
        }
    """
    result = pipeline.query(question["question"], top_k=top_k)
    sources = result.get("sources", [])

    # 命中预期文档
    expected_doc = question.get("expected_doc", "")
    hit_expected = False
    if expected_doc:
        expected_base = expected_doc.rsplit(".", 1)[0]
        hit_expected = any(
            expected_doc == s["metadata"].get("filename", "")
            or s["metadata"].get("filename", "") == expected_base
            or s["metadata"]
            .get("filename", "")
            .startswith(expected_base.split("_", 1)[-1] if "_" in expected_base else expected_base)
            for s in sources
        )

    # 分数
    scores = [s.get("_vector_score", s.get("score", 0)) for s in sources]
    top_score = scores[0] if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "question_id": question.get("id", ""),
        "question": question["question"],
        "hit_expected_doc": hit_expected,
        "num_retrieved": len(sources),
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "scores": scores,
        "has_relevant": result.get("relevance", {}).get("is_relevant", False),
        "is_refusal": result.get("is_refusal", False),
    }


def evaluate_generation(pipeline, question: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """对单条 Golden 问题执行回答评估（含关键词检查）

    Args:
        pipeline: RAGPipeline 实例
        question: Golden 题目 dict
        top_k: 检索数量

    Returns:
        {
            "question_id": str,
            "answer": str,
            "missing_keywords": [str, ...],     # 缺失的关键词
            "keyword_hit_rate": float,          # 0-1
            "has_refusal": bool,                # 是否拒答
        }
    """
    result = pipeline.query(question["question"], top_k=top_k)
    answer = result.get("answer", "")
    keywords = question.get("expected_answer_keywords", [])

    missing = [kw for kw in keywords if kw not in answer]
    hit_rate = (len(keywords) - len(missing)) / len(keywords) if keywords else 1.0

    return {
        "question_id": question.get("id", ""),
        "answer": answer,
        "missing_keywords": missing,
        "keyword_hit_rate": round(hit_rate, 2),
        "has_refusal": result.get("is_refusal", False),
    }


def format_golden_summary(retrieval_results: list[dict[str, Any]]) -> str:
    """格式化检索评估摘要"""
    total = len(retrieval_results)
    exact_questions = [r for r in retrieval_results if r.get("question_id", "").startswith("exact_")]
    if not exact_questions:
        return "（无精确匹配题）"

    hits = sum(1 for r in exact_questions if r["hit_expected_doc"])
    avg_score = sum(r["top_score"] for r in retrieval_results) / total if total else 0

    lines = [
        f"📊 Golden 数据集评估: {hits}/{len(exact_questions)} 命中预期文档",
        f"   平均 Top Score: {avg_score:.4f}",
    ]
    return "\n".join(lines)

"""
测试问题集
分为三类四种难度：
  - exact_match:    知识库中存在明确答案，且可定位到预期文档
  - cross_doc:      需要综合多个文档的信息
  - out_of_knowledge: 知识库范围外的问题，用于评估拒答准确率

每道题标记 difficulty（easy / medium / hard），支持按难度层级切片分析。
"""

import json
import os
from typing import Any

# ── 内建测试问题集 ──────────────────────────────────

BUILTIN_QUESTIONS = [
    # ══════════════════════════════════════════════════
    #  精确匹配（expected_doc 应出现在检索结果中）
    # ══════════════════════════════════════════════════
    # ── easy: 检索词高度匹配文件名 ──
    {
        "id": "exact_01",
        "question": "Transformer在医学影像分析中有哪些应用？",
        "category": "exact_match",
        "expected_doc": "06_Transformer在医学影像中的应用.md",
        "difficulty": "easy",
    },
    {
        "id": "exact_02",
        "question": "CT影像肺结节的检测流程是怎样的？",
        "category": "exact_match",
        "expected_doc": "02_CT影像肺结节检测技术方案.md",
        "difficulty": "easy",
    },
    {
        "id": "exact_03",
        "question": "U-Net网络的核心创新是什么？",
        "category": "exact_match",
        "expected_doc": "03_U-Net医学图像分割论文笔记.md",
        "difficulty": "easy",
    },
    {
        "id": "exact_04",
        "question": "公司数据安全的L3级别数据包括哪些？",
        "category": "exact_match",
        "expected_doc": "05_企业数据安全管理制度.md",
        "difficulty": "medium",
    },
    # ── medium: 需语义匹配，文件名不含完整问题词 ──
    {
        "id": "exact_05",
        "question": "CTPA上如何区分急性和慢性肺栓塞？",
        "category": "exact_match",
        "expected_doc": "PE的病理分型：急性 vs. 慢性.md",
        "difficulty": "medium",
    },
    {
        "id": "exact_06",
        "question": "肺栓塞CTPA的直接征象有哪些？",
        "category": "exact_match",
        "expected_doc": "CTPA影像表现：从直接到间接.md",
        "difficulty": "medium",
    },
    {
        "id": "exact_07",
        "question": "sPESI评分包含哪些评估项目？",
        "category": "exact_match",
        "expected_doc": "临床危险分层：理解sPESI评分.md",
        "difficulty": "medium",
    },
    # ── hard: 需要检索到具体内容片段而非泛泛匹配 ──
    {
        "id": "exact_08",
        "question": "D-二聚体在肺栓塞诊断中的年龄校正临界值是什么？",
        "category": "exact_match",
        "expected_doc": "诊断流程指南：从怀疑到确诊.md",
        "difficulty": "hard",
    },
    {
        "id": "exact_09",
        "question": "急性肺栓塞导致右心功能改变的病理机制是什么？",
        "category": "exact_match",
        "expected_doc": "PE的病理分型：急性 vs. 慢性.md",
        "difficulty": "hard",
    },
    {
        "id": "exact_10",
        "question": "YEARS模型在肺栓塞诊断中如何应用？",
        "category": "exact_match",
        "expected_doc": "诊断流程指南：从怀疑到确诊.md",
        "difficulty": "hard",
    },
    # ══════════════════════════════════════════════════
    #  跨文档综合（需要检索多个文档的片段）
    # ══════════════════════════════════════════════════
    {
        "id": "cross_01",
        "question": "医学影像AI系统的技术栈包括哪些？",
        "category": "cross_doc",
        "expected_doc": "",
        "difficulty": "medium",
    },
    {
        "id": "cross_02",
        "question": "肺结节检测使用哪些深度学习模型？",
        "category": "cross_doc",
        "expected_doc": "",
        "difficulty": "medium",
    },
    {
        "id": "cross_03",
        "question": "肺栓塞的诊断流程是怎样的？从临床评估到影像确诊。",
        "category": "cross_doc",
        "expected_doc": "",
        "difficulty": "hard",
    },
    {
        "id": "cross_04",
        "question": "肺栓塞如何根据危险分层选择治疗方案？",
        "category": "cross_doc",
        "expected_doc": "",
        "difficulty": "hard",
    },
    {
        "id": "cross_05",
        "question": "CTPA影像上如何全面评估肺栓塞的严重程度？",
        "category": "cross_doc",
        "expected_doc": "",
        "difficulty": "hard",
    },
    # ══════════════════════════════════════════════════
    #  知识库外（期望拒答或给出低分）
    # ══════════════════════════════════════════════════
    # ── 非医学领域（明确无关） ──
    {
        "id": "ood_01",
        "question": "2025年全球经济增长率是多少？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "easy",
    },
    {
        "id": "ood_02",
        "question": "如何配置Kubernetes集群的RBAC权限？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "easy",
    },
    {
        "id": "ood_03",
        "question": "Python中如何使用async/await进行异步编程？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "medium",
    },
    {
        "id": "ood_04",
        "question": "React 18的主要新特性有哪些？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "medium",
    },
    # ── 医学相关但不在本知识库中（更难的拒答场景） ──
    {
        "id": "ood_05",
        "question": "COVID-19对心血管系统有哪些长期影响？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "hard",
    },
    {
        "id": "ood_06",
        "question": "糖尿病肾病患者的血压控制目标是多少？",
        "category": "out_of_knowledge",
        "expected_doc": "",
        "difficulty": "hard",
    },
]


def get_test_questions() -> list[dict[str, Any]]:
    """获取默认测试问题集

    优先从 tests/test_questions.json 加载（Golden 数据集），
    不存在时回退到内建常量（保持向后兼容）。
    """
    json_path = os.path.join(os.path.dirname(__file__), "..", "tests", "test_questions.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return list(BUILTIN_QUESTIONS)


def load_questions_from_json(path: str) -> list[dict[str, Any]]:
    """从 JSON 文件加载测试问题

    JSON 格式:
        [
            {"question": "...", "category": "exact_match", "expected_doc": "..."},
            ...
        ]
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_questions_by_category(questions: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """按类别筛选问题"""
    return [q for q in questions if q.get("category") == category]


def get_questions_by_difficulty(questions: list[dict[str, Any]], difficulty: str) -> list[dict[str, Any]]:
    """按难度筛选问题"""
    return [q for q in questions if q.get("difficulty") == difficulty]


def print_question_summary(questions: list[dict[str, Any]]) -> None:
    """打印问题集摘要"""
    from collections import Counter

    cats = Counter(q.get("category", "unknown") for q in questions)
    diffs = Counter(q.get("difficulty", "unknown") for q in questions)
    print(f"📋 测试问题集: {len(questions)} 题")
    for cat, count in cats.items():
        print(f"   - {cat}: {count} 题")
    diff_str = ", ".join(f"{d}={c}" for d, c in sorted(diffs.items()))
    print(f"   - 难度分布: {diff_str}")

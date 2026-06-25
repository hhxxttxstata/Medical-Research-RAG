"""
Golden 数据集回归测试

每次修改检索/生成逻辑后运行：
  pytest tests/test_eval_golden.py -x -v

需要 ChromaDB 知识库已初始化。
标记为 @pytest.mark.integration，默认不跑。
"""

import pytest

from .eval_utils import load_golden_questions

# ── 共享 Pipeline（模块级别，只初始化一次） ──────────


@pytest.fixture(scope="module")
def pipeline():
    """初始化 RAGPipeline"""
    from src.rag_pipeline import RAGPipeline

    pl = RAGPipeline(top_k=5)
    # 确保知识库已初始化
    count = pl.vector_store.count()
    if count == 0:
        pl.initialize_knowledge_base()
    yield pl
    pl.close()


# ── 精确匹配：检索命中预期文档 ──────────────────────


@pytest.mark.integration
class TestGoldenExactMatch:
    """精确匹配题应命中预期文档"""

    @pytest.mark.parametrize("question", load_golden_questions("exact_match"))
    def test_hits_expected_doc(self, pipeline, question):
        """检索结果中应包含 expected_doc"""
        from .eval_utils import evaluate_retrieval

        result = evaluate_retrieval(pipeline, question, top_k=5)
        assert result["hit_expected_doc"], f"预期文档 '{question.get('expected_doc')}' 未命中"


# ── 领域外：应拒答 ───────────────────────────────────


@pytest.mark.integration
class TestGoldenOOD:
    """领域外问题应拒答"""

    @pytest.mark.parametrize("question", load_golden_questions("out_of_knowledge"))
    def test_ood_refusal(self, pipeline, question):
        """OOD 题应标记为 is_refusal=True 或 is_relevant=False"""
        from .eval_utils import evaluate_retrieval

        result = evaluate_retrieval(pipeline, question, top_k=5)
        assert result["is_refusal"] or not result["has_relevant"], f"OOD 问题 '{question['question'][:30]}' 未被拒答"


# ── 回答关键词检查（所有题目） ──────────────────────


@pytest.mark.integration
class TestGoldenKeywords:
    """回答应包含期望关键词（仅限非 OOD 题目）"""

    @pytest.mark.parametrize("question", load_golden_questions())
    def test_answer_contains_keywords(self, pipeline, question):
        """回答应包含 expected_answer_keywords"""
        keywords = question.get("expected_answer_keywords", [])
        if not keywords:
            pytest.skip("无关键词约束")

        from .eval_utils import evaluate_generation

        result = evaluate_generation(pipeline, question, top_k=5)
        missing = result["missing_keywords"]
        assert not missing, f"关键词缺失: {missing} (命中率 {result['keyword_hit_rate']:.0%})"


# ── 快捷验证：不依赖 ChromaDB ──────────────────────


class TestGoldenDataIntegrity:
    """Golden 数据集自身完整性校验（不依赖知识库）"""

    def test_all_questions_have_ids(self):
        """每题都有 id"""
        questions = load_golden_questions()
        for q in questions:
            assert q.get("id"), f"题目缺失 id: {q['question'][:30]}"

    def test_ids_are_unique(self):
        """id 不重复"""
        questions = load_golden_questions()
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), "存在重复 id"

    def test_all_questions_have_categories(self):
        """每题都有 category"""
        questions = load_golden_questions()
        valid_categories = {"exact_match", "cross_doc", "out_of_knowledge"}
        for q in questions:
            assert q.get("category") in valid_categories, f"无效 category: {q.get('category')} in {q['id']}"

    def test_all_questions_have_difficulty(self):
        """每题都有 difficulty"""
        questions = load_golden_questions()
        valid_difficulties = {"easy", "medium", "hard"}
        for q in questions:
            assert q.get("difficulty") in valid_difficulties, f"无效 difficulty: {q.get('difficulty')} in {q['id']}"

    def test_keyword_count(self):
        """exact_match 和 cross_doc 题应有 2-5 个关键词"""
        questions = [q for q in load_golden_questions() if q["category"] in ("exact_match", "cross_doc")]
        for q in questions:
            kws = q.get("expected_answer_keywords", [])
            assert 1 <= len(kws) <= 5, f"{q['id']}: 关键词数量 {len(kws)} 超出范围 [1, 5]"

    def test_ood_keywords_empty(self):
        """OOD 题的关键词应为空"""
        questions = load_golden_questions("out_of_knowledge")
        for q in questions:
            assert q.get("expected_answer_keywords") == [], f"OOD 题 {q['id']} 不应有关键词"

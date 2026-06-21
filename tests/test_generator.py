"""
生成模块单元测试

测试策略：
  - compute_relevance：纯数学计算，重点测边界阈值
  - build_rag_prompt：验证 prompt 结构完整性
  - validate_citations：验证引用检测逻辑
  - LLMGenerator：不测真实 API 调用（需网络），只测辅助方法
"""

from src.generator import (
    LLMGenerator,
    _extract_query_from_prompt,
    build_rag_prompt,
    compute_relevance,
    validate_citations,
)


class TestComputeRelevance:
    """多因子相关性评分测试"""

    def test_high_score_relevant(self, sample_chunks):
        """高相似度应判定为 relevant"""
        result = compute_relevance("肺栓塞诊断", sample_chunks)
        assert result["is_relevant"] is True
        assert result["top1_score"] == 0.85
        # avg_score 内部被 round 到 4 位小数
        expected_avg = round((0.85 + 0.72 + 0.45) / 3, 4)
        assert result["avg_score"] == expected_avg

    def test_low_score_irrelevant(self, sample_chunks_low_score):
        """低相似度 + 低重叠应判定为 irrelevant"""
        result = compute_relevance("天气如何", sample_chunks_low_score)
        assert result["is_relevant"] is False
        assert result["top1_score"] < 0.10

    def test_empty_chunks_irrelevant(self):
        """空检索结果应返回 not relevant"""
        result = compute_relevance("任何问题", [])
        assert result["is_relevant"] is False
        assert result["reason"] == "检索结果为空"

    def test_top1_above_threshold(self):
        """top1_score 超过 0.50 即使综合分低也判定为相关"""
        chunks = [
            {"id": "c1", "text": "肺栓塞" * 50, "metadata": {}, "score": 0.60},
        ]
        result = compute_relevance("肺栓塞", chunks)
        assert result["is_relevant"] is True

    def test_overlap_boosts_relevance(self):
        """文本重叠可补偿低语义分"""
        chunks = [
            {
                "id": "c1",
                "text": "肺栓塞是一种危急重症，CTPA是诊断肺栓塞的金标准。",
                "metadata": {},
                "score": 0.20,
            },
        ]
        result = compute_relevance("肺栓塞CTPA诊断", chunks)
        # overlap 足够高时综合分应达标
        if result["overlap"] > 0.05:
            assert result["is_relevant"] is True


class TestBuildRagPrompt:
    """Prompt 构建测试"""

    def test_prompt_contains_required_sections(self, sample_chunks):
        """生成的 prompt 应包含核心部分"""
        prompt, source_map, relevance = build_rag_prompt("肺栓塞是什么", sample_chunks)
        assert "## 参考文档" in prompt
        assert "## 用户问题" in prompt
        assert "## 回答" in prompt
        assert "肺栓塞是什么" in prompt

    def test_source_map_contains_all_chunks(self, sample_chunks):
        """source_map 应包含所有传入的 chunk"""
        _, source_map, _ = build_rag_prompt("肺栓塞是什么", sample_chunks)
        assert len(source_map) == len(sample_chunks)
        for key in source_map:
            assert "filename" in source_map[key]
            assert "score" in source_map[key]

    def test_prompt_with_relevance_param(self, sample_chunks):
        """传入 relevance 参数不应影响 prompt 结构"""
        relevance = {"is_relevant": True, "top1_score": 0.9, "avg_score": 0.8, "overlap": 0.1, "reason": "test"}
        prompt, source_map, returned_rel = build_rag_prompt("肺栓塞是什么", sample_chunks, relevance)
        assert "## 参考文档" in prompt
        assert "## 用户问题" in prompt
        assert returned_rel == relevance

    def test_prompt_empty_chunks(self):
        """空 chunk 列表时 prompt 仍应包含必要部分"""
        prompt, source_map, relevance = build_rag_prompt("肺栓塞是什么", [])
        assert "## 参考文档" in prompt
        assert "（无参考文档）" in prompt
        assert source_map == {}


class TestValidateCitations:
    """引用验证测试"""

    def test_all_valid(self):
        """所有引用 [1][2] 都在 source_map 中"""
        answer = "根据资料[1]，肺栓塞需要及时治疗[2]。"
        source_map = {"1": {"filename": "a.md"}, "2": {"filename": "b.md"}}
        result = validate_citations(answer, source_map)
        assert result["has_invalid_citations"] is False
        assert "1" in result["cited_valid"]
        assert "2" in result["cited_valid"]

    def test_invalid_ref_detected(self):
        """引用 [99] 不在 source_map 中应标记为 invalid"""
        answer = "根据资料[1]和[99]的结论。"
        source_map = {"1": {"filename": "a.md"}}
        result = validate_citations(answer, source_map)
        assert result["has_invalid_citations"] is True
        assert "99" in result["cited_invalid"]

    def test_no_citations(self):
        """回答中无引用，返回空 valid 列表"""
        answer = "这是一个没有引用的回答。"
        source_map = {"1": {"filename": "a.md"}}
        result = validate_citations(answer, source_map)
        assert result["cited_valid"] == []
        assert result["cited_invalid"] == []

    def test_unused_sources_detected(self):
        """source_map 中有但回答未引用的来源被标记"""
        answer = "根据资料[1]。"
        source_map = {"1": {"filename": "a.md"}, "2": {"filename": "b.md"}}
        result = validate_citations(answer, source_map)
        assert "2" in result["unused"]
        assert result["has_unused_sources"] is True

    def test_empty_source_map(self):
        """空的 source_map 时不应报错"""
        answer = "无引用回答。"
        result = validate_citations(answer, {})
        assert result["cited_valid"] == []
        assert result["cited_invalid"] == []
        assert result["has_invalid_citations"] is False


class TestExtractQueryFromPrompt:
    """从 prompt 中提取用户问题"""

    def test_extract_query(self):
        prompt = "## 用户问题\n肺栓塞怎么治疗？\n## 回答\n"
        assert _extract_query_from_prompt(prompt) == "肺栓塞怎么治疗？"

    def test_extract_query_inline(self):
        prompt = "## 用户问题 肺栓塞\n## 回答"
        assert _extract_query_from_prompt(prompt) == "肺栓塞"

    def test_extract_query_not_found(self):
        assert _extract_query_from_prompt("无问题标记") == ""


class TestLLMGenerator:
    """LLMGenerator 辅助方法测试（不调用真实 API）"""

    def test_invalid_api_key_empty(self):
        """空字符串应判定为无效 key"""
        gen = LLMGenerator(api_key="")
        assert gen._is_valid_api_key("") is False

    def test_invalid_api_key_placeholder(self):
        """占位符 key 应判定为无效"""
        gen = LLMGenerator(api_key="sk-your-api-key-here")
        assert gen._is_valid_api_key("sk-your-api-key-here") is False

    def test_valid_api_key(self):
        """正常 key 应判定为有效"""
        gen = LLMGenerator(api_key="sk-xxxxxxxxxxxx")
        assert gen._is_valid_api_key("sk-xxxxxxxxxxxx") is True

    def test_refusal_response_format(self):
        """拒答回复应包含结论、依据等标准格式"""
        gen = LLMGenerator(api_key="")
        response = gen._build_refusal_response("测试问题", "无相关文档")
        assert "**结论：**" in response
        assert "**依据：**" in response
        assert "测试问题" in response
        assert "无相关文档" in response

    def test_fallback_response_with_sources(self):
        """无 API 时的兜底回答应列出引用来源"""
        gen = LLMGenerator(api_key="")
        source_map = {
            "1": {"filename": "test.md", "page": "1", "paragraph": "1-2", "score": 0.9, "text_preview": "测试"},
        }
        response = gen._fallback_structured_response("prompt", source_map)
        assert "检索到 1 篇相关文档" in response
        assert "test.md" in response

    def test_fallback_response_empty_sources(self):
        """无 API + 无 sources 时应返回拒答回复"""
        gen = LLMGenerator(api_key="")
        response = gen._fallback_structured_response("prompt\n## 用户问题\n测试问题\n", {})
        assert "**结论：**" in response

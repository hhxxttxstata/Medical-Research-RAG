"""
三级召回流水线单元测试

覆盖：
  - Query Rewriting（解析、降级、开关）
  - Reranker（解析、降级、开关）
  - 完整三级串联（Rewrite → Hybrid → Rerank）
  - 向后兼容
"""

import json

from src.retriever import Retriever

# ═══════════════════════════════════════════════════════════════
#  测试辅助
# ═══════════════════════════════════════════════════════════════


class MockEmbeddingProvider:
    """模拟 Embedding，支持 prefix 参数"""

    def embed(self, texts: list[str], prefix: str | None = None) -> list[list[float]]:
        return [[0.01] * 384 for _ in texts]


class MockVectorStore:
    """模拟向量存储"""

    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def similarity_search(self, query_embedding, top_k=5):
        return self.chunks[:top_k]

    def get_all_documents(self):
        return self.chunks

    def count(self):
        return len(self.chunks)


class MockGenerator:
    """可控的 LLM 模拟"""

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.call_history: list = []

    def chat(self, messages, temperature=0.0, max_tokens=256) -> str:
        self.call_history.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        for msg in reversed(messages):
            if msg["role"] == "user":
                key = msg["content"][:50]
                for pattern, resp in self.responses.items():
                    if pattern in key:
                        return resp
        return self.responses.get("__default__", "默认回复")


def make_chunks(count: int, base_text: str = "肺栓塞诊断相关内容。") -> list[dict]:
    """生成模拟 chunk"""
    return [
        {
            "id": f"chunk_{i}",
            "text": f"{base_text} 第{i}段内容。",
            "metadata": {"filename": "test.md", "page": i},
            "score": 0.9 - i * 0.05,
        }
        for i in range(count)
    ]


# ═══════════════════════════════════════════════════════════════
#  一、Query Rewriting 测试
# ═══════════════════════════════════════════════════════════════


class TestQueryRewriting:
    """Query Rewriting 单元测试"""

    def test_disabled(self):
        """enable_rewrite=False 时 _can_rewrite 返回 False"""
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=MockGenerator(),
            enable_rewrite=False,
            enable_reranker=False,
        )
        assert hr._can_rewrite() is False

    def test_no_generator(self):
        """generator=None 时返回 [原始query]"""
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=None,
            enable_rewrite=True,
            enable_reranker=False,
        )
        assert hr._can_rewrite() is False
        queries, ood = hr._rewrite_query("测试查询")
        assert queries == ["测试查询"]
        assert ood is False

    def test_parse_multi_line(self):
        """多行输出解析为多条 query"""
        gen = MockGenerator(
            responses={
                "原始问题：看看CT": "肺栓塞CTPA诊断\n肺栓塞影像表现\nCT肺动脉造影",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        queries, _ = hr._rewrite_query("看看CT")
        assert len(queries) == 3
        assert "CTPA" in queries[0] or "肺栓塞" in queries[0]

    def test_parse_single_line(self):
        """单行输出解析为一条 query"""
        gen = MockGenerator(
            responses={
                "原始问题：什么是肺栓塞": "肺栓塞定义 病因 临床表现",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        queries, _ = hr._rewrite_query("什么是肺栓塞")
        assert len(queries) == 1

    def test_parse_strips_numbering(self):
        """解析应去掉编号前缀"""
        gen = MockGenerator(
            responses={
                "原始问题：诊断方法": "1. 肺栓塞诊断方法\n2. CTPA影像学检查",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        queries, _ = hr._rewrite_query("诊断方法")
        assert "1." not in queries[0]
        assert "2." not in queries[1]

    def test_llm_failure_fallback(self):
        """LLM 调用失败时回退到 [原始query]"""

        class FailingGenerator:
            def chat(self, messages, temperature=0.0, max_tokens=256) -> str:
                raise RuntimeError("API fail")

        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=FailingGenerator(),
            enable_reranker=False,
        )
        queries, ood = hr._rewrite_query("原始查询")
        assert queries == ["原始查询"]
        assert ood is False

    def test_empty_response_fallback(self):
        """LLM 返回空内容时回退"""
        gen = MockGenerator(
            responses={
                "__default__": "",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        queries, _ = hr._rewrite_query("test")
        assert queries == ["test"]

    def test_prompt_contains_query(self):
        """验证 prompt 中包含用户 query"""
        gen = MockGenerator(
            responses={
                "__default__": "改写后的查询",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        hr._rewrite_query("我的测试查询")
        assert len(gen.call_history) >= 1
        user_msg = gen.call_history[0]["messages"][1]["content"]
        assert "我的测试查询" in user_msg

    def test_rewrite_temperature_zero(self):
        """rewrite temperature 应为 0"""
        gen = MockGenerator(
            responses={
                "__default__": "改写结果",
            }
        )
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_reranker=False,
        )
        hr._rewrite_query("test")
        assert gen.call_history[0]["temperature"] == 0.0

    def test_parse_removes_empty_lines(self):
        """空行应被过滤"""
        lines = "\n\n\n肺栓塞诊断\n\n\nCT表现\n\n"
        result = Retriever._parse_rewrite_response(lines)
        assert result is not None
        assert len(result) == 2
        assert "肺栓塞诊断" in result


# ═══════════════════════════════════════════════════════════════
#  二、Reranker 测试
# ═══════════════════════════════════════════════════════════════


class TestReranker:
    """Reranker 单元测试"""

    def test_disabled(self):
        """enable_reranker=False 时跳过"""
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=MockGenerator(),
            enable_rewrite=False,
            enable_reranker=False,
        )
        assert hr._can_rerank() is False

    def test_no_generator(self):
        """generator=None 时跳过"""
        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=None,
            enable_rewrite=False,
            enable_reranker=True,
        )
        assert hr._can_rerank() is False

    def test_parse_json_scores(self):
        """解析标准 JSON 评分列表"""
        chunks = make_chunks(3)
        response = json.dumps(
            [
                {"id": "chunk_0", "relevance": 9, "reason": "直接相关"},
                {"id": "chunk_1", "relevance": 3, "reason": "部分相关"},
                {"id": "chunk_2", "relevance": 7, "reason": "相关"},
            ]
        )
        scores = Retriever._parse_rerank_response(response, 3)
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == 9.0
        assert scores[1] == 3.0
        assert scores[2] == 7.0

    def test_parse_with_padding(self):
        """评分数量不足时用 0 填充"""
        response = json.dumps([{"relevance": 8}])
        scores = Retriever._parse_rerank_response(response, 5)
        assert scores is not None
        assert len(scores) == 5
        assert scores[0] == 8.0
        assert scores[-1] == 0.0

    def test_parse_markdown_fence(self):
        """能处理 ```json 包裹"""
        response = '```json\n[{"relevance": 8}, {"relevance": 5}]\n```'
        scores = Retriever._parse_rerank_response(response, 2)
        assert scores is not None
        assert scores[0] == 8.0

    def test_parse_invalid_fallback(self):
        """无效输出返回 None"""
        assert Retriever._parse_rerank_response("不是JSON", 3) is None
        assert Retriever._parse_rerank_response("", 3) is None

    def test_rerank_sorts_by_score(self):
        """reranker 应高分在前"""

        # Cross-encoder reranker 路径（LLM-as-reranker 已被 CrossEncoderReranker 替代）
        class MockReranker:
            model_ready = True

            def rerank(self, query, chunks, top_k):
                for i, c in enumerate(chunks):
                    c["_rerank_score"] = [3.0, 9.0, 6.0][i]
                chunks.sort(key=lambda x: x["_rerank_score"], reverse=True)
                return chunks[:top_k]

        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=MockGenerator(),
            enable_rewrite=False,
            enable_reranker=True,
            reranker=MockReranker(),
        )
        chunks = make_chunks(3)
        reranked = hr._rerank("测试", chunks, top_k=3)
        assert reranked[0]["_rerank_score"] == 9.0
        assert reranked[2]["_rerank_score"] == 3.0

    def test_rerank_failure_fallback(self):
        """reranker 调用失败时回退到原始顺序"""

        class FailingGenerator:
            def chat(self, messages, temperature=0.0, max_tokens=1024) -> str:
                raise RuntimeError("API fail")

        hr = Retriever(
            vector_store=MockVectorStore(),
            embedding_provider=MockEmbeddingProvider(),
            generator=FailingGenerator(),
            enable_rewrite=False,
            enable_reranker=True,
        )
        chunks = make_chunks(5)
        reranked = hr._rerank("测试", chunks, top_k=3)
        assert len(reranked) == 3


# ═══════════════════════════════════════════════════════════════
#  三、完整流水线集成测试
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    """三级召回流水线集成测试"""

    def test_retrieve_basic(self):
        """无 rewrite/rerank 时，基础混合检索应正常工作"""
        chunks = make_chunks(10)
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=None,
            enable_rewrite=False,
            enable_reranker=False,
        )
        results = hr.retrieve("肺栓塞诊断", top_k=3)
        assert len(results) == 3
        assert all("_rrf_score" not in r for r in results)  # 内部字段已清理

    def test_retrieve_backward_compatible(self):
        """不传 generator 时表现与旧版一致"""
        chunks = make_chunks(5)
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
        )
        # generator 默认为 None → rewrite 和 rerank 自动跳过
        results = hr.retrieve("肺栓塞诊断", top_k=3)
        assert len(results) == 3

    def test_rewrite_with_hybrid(self):
        """rewrite + hybrid 串联：多条查询去重合并"""
        gen = MockGenerator(
            responses={
                "原始问题：CT检查": "肺栓塞CTPA\n肺栓塞影像诊断",
            }
        )
        chunks = make_chunks(20, "肺栓塞诊断相关内容。")
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_rewrite=True,
            enable_reranker=False,
        )
        results = hr.retrieve("CT检查", top_k=5)
        assert len(results) == 5

    def test_rerank_with_hybrid(self):
        """hybrid + rerank 串联：reranker 按分排序"""
        gen = MockGenerator(
            responses={
                "__default__": json.dumps([{"relevance": 9 - i, "reason": "auto"} for i in range(10)]),
            }
        )
        chunks = make_chunks(15, "肺栓塞诊断相关内容。")
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_rewrite=False,
            enable_reranker=True,
        )
        results = hr.retrieve("肺栓塞诊断", top_k=3)
        # reranker 的 score 应覆盖 RRF 分数
        assert len(results) == 3
        # score 应反映 reranker 评分（9, 8, 7... 降序）
        assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]

    def test_full_pipeline(self):
        """三级全开：rewrite + hybrid + rerank"""
        gen = MockGenerator(
            responses={
                "原始问题：看看这个CT": "肺栓塞CTPA诊断\n肺栓塞影像表现",
                "__default__": json.dumps([{"relevance": 9 - i, "reason": "auto"} for i in range(15)]),
            }
        )
        chunks = make_chunks(30, "肺栓塞诊断相关内容。")
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_rewrite=True,
            enable_reranker=True,
        )
        results = hr.retrieve("看看这个CT有没有肺栓塞", top_k=3)
        assert len(results) == 3
        # score 是 reranker 分
        assert results[0]["score"] > results[1]["score"]

    def test_top_k_more_than_available(self):
        """请求结果多于可用 chunk 时返回全部"""
        chunks = make_chunks(2)
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=None,
            enable_rewrite=False,
            enable_reranker=False,
        )
        results = hr.retrieve("test", top_k=10)
        assert len(results) == 2

    def test_retrieve_original_query_always_included(self):
        """rewrite 返回的查询中，原始 query 应始终是选项之一（通过 fallback）"""
        gen = MockGenerator(
            responses={
                "__default__": "",  # 空响应 → fallback 到 [原始query]
            }
        )
        chunks = make_chunks(5)
        hr = Retriever(
            vector_store=MockVectorStore(chunks),
            embedding_provider=MockEmbeddingProvider(),
            generator=gen,
            enable_rewrite=True,
            enable_reranker=False,
        )
        results = hr.retrieve("原始查询内容", top_k=3)
        assert len(results) == 3

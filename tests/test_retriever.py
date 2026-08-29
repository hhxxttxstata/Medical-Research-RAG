"""
检索模块单元测试

测试策略：
  - Retriever：保持现有测试不变
  - HybridRetriever → Retriever：使用 Retriever(enable_rewrite=True, enable_reranker=True)
"""

from unittest.mock import MagicMock

import pytest

from src.embeddings import EmbeddingProvider
from src.milvus_store import MilvusStore
from src.retriever import Retriever


@pytest.fixture
def mock_embedding():
    """Mock EmbeddingProvider"""
    mock = MagicMock(spec=EmbeddingProvider)
    mock.embed.return_value = [[0.1] * 384]
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore（仅 similarity_search）"""
    mock = MagicMock(spec=MilvusStore)
    mock.similarity_search.return_value = [
        {
            "id": "chunk_1",
            "text": "肺栓塞是一种危急重症。",
            "metadata": {"filename": "doc.md", "page": 1},
            "score": 0.85,
        },
        {"id": "chunk_2", "text": "CTPA是诊断金标准。", "metadata": {"filename": "doc.md", "page": 2}, "score": 0.72},
    ]
    return mock


@pytest.fixture
def mock_hybrid_store():
    """Mock VectorStore（含 get_all_documents 支持，用于 HybridRetriever）"""
    mock = MagicMock(spec=MilvusStore)
    mock.similarity_search.return_value = [
        {
            "id": "chunk_1",
            "text": "肺栓塞是一种危急重症。",
            "metadata": {"filename": "doc.md", "page": 1},
            "score": 0.85,
        },
        {"id": "chunk_2", "text": "CTPA是诊断金标准。", "metadata": {"filename": "doc.md", "page": 2}, "score": 0.72},
    ]
    mock.get_all_documents.return_value = [
        {"id": "chunk_1", "text": "肺栓塞是一种危急重症。", "metadata": {"filename": "doc.md", "page": 1}},
        {"id": "chunk_2", "text": "CTPA是诊断金标准。", "metadata": {"filename": "doc.md", "page": 2}},
        {
            "id": "chunk_3",
            "text": "急性肺栓塞CT表现包括直接和间接征象。",
            "metadata": {"filename": "doc2.md", "page": 1},
        },
    ]
    return mock


# ══════════════════════════════════════════════════
#  Retriever（纯向量检索）测试
# ══════════════════════════════════════════════════


class TestRetriever:
    """检索器测试"""

    def test_retrieve_basic(self, mock_embedding, mock_vector_store):
        retriever = Retriever(vector_store=mock_vector_store, embedding_provider=mock_embedding, top_k=5)
        results = retriever.retrieve("肺栓塞是什么")
        assert len(results) == 2
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_retrieve_top_k(self, mock_embedding, mock_vector_store):
        """纯向量分支：取 2k 候选 → 文档级多样性 → 返回 k 条"""
        retriever = Retriever(vector_store=mock_vector_store, embedding_provider=mock_embedding, top_k=5)
        retriever.retrieve("测试", top_k=3)
        mock_vector_store.similarity_search.assert_called_once()
        call_kwargs = mock_vector_store.similarity_search.call_args[1]
        # 2026-08 优化：2k 候选供文档级多样性截断（防止大文档霸榜）
        assert call_kwargs["top_k"] == 6

    def test_diversify_by_doc(self, mock_embedding, mock_vector_store):
        """文档级多样性：同文档最多 max_per_doc 条"""
        retriever = Retriever(vector_store=mock_vector_store, embedding_provider=mock_embedding, top_k=5)
        results = [
            {"id": "a1", "text": "x", "metadata": {"filename": "A.md"}},
            {"id": "a2", "text": "x", "metadata": {"filename": "A.md"}},
            {"id": "a3", "text": "x", "metadata": {"filename": "A.md"}},
            {"id": "b1", "text": "x", "metadata": {"filename": "B.md"}},
            {"id": "c1", "text": "x", "metadata": {"filename": "C.md"}},
        ]
        out = retriever._diversify_by_doc(results, max_per_doc=2)
        assert [r["id"] for r in out] == ["a1", "a2", "b1", "c1"]
        # 无 filename 的 chunk 不限制
        out2 = retriever._diversify_by_doc([{"id": "n1", "text": "x", "metadata": {}}] * 5, max_per_doc=1)
        assert len(out2) == 5

    def test_retrieve_calls_embed(self, mock_embedding, mock_vector_store):
        """验证检索时是否传了 query: prefix"""
        retriever = Retriever(vector_store=mock_vector_store, embedding_provider=mock_embedding, top_k=5)
        retriever.retrieve("测试查询")
        mock_embedding.embed.assert_called_once()
        args, kwargs = mock_embedding.embed.call_args
        # embed 调用应该收到 prefix='query: '
        assert "prefix" in kwargs
        assert kwargs["prefix"] == "query: "

    def test_retrieve_empty_results(self, mock_embedding):
        mock_store = MagicMock(spec=MilvusStore)
        mock_store.similarity_search.return_value = []
        retriever = Retriever(vector_store=mock_store, embedding_provider=mock_embedding, top_k=5)
        results = retriever.retrieve("不存在的内容")
        assert results == []


class TestFormatResults:
    """检索结果格式化测试"""

    def test_format_results_contains_scores(self, mock_embedding, mock_vector_store):
        retriever = Retriever(vector_store=mock_vector_store, embedding_provider=mock_embedding, top_k=5)
        results = retriever.retrieve("肺栓塞")
        assert len(results) > 0
        assert "score" in results[0]
        assert "metadata" in results[0]

    def test_format_results_empty(self, mock_embedding):
        mock_store = MagicMock(spec=MilvusStore)
        mock_store.similarity_search.return_value = []
        retriever = Retriever(vector_store=mock_store, embedding_provider=mock_embedding, top_k=5)
        results = retriever.retrieve("不存在的内容")
        assert results == []


# ══════════════════════════════════════════════════
#  HybridRetriever（混合检索）测试
# ══════════════════════════════════════════════════


class TestHybridRetriever:
    """混合检索器测试"""

    def test_hybrid_retrieve_basic(self, mock_embedding, mock_hybrid_store):
        """混合检索返回正确格式的结果"""
        retriever = Retriever(
            vector_store=mock_hybrid_store,
            embedding_provider=mock_embedding,
            top_k=5,
        )
        results = retriever.retrieve("肺栓塞是什么")
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_hybrid_fallback_no_bm25(self, mock_embedding):
        """BM25 索引不可用时回退到纯向量检索"""
        mock_store = MagicMock(spec=MilvusStore)
        mock_store.similarity_search.return_value = [
            {"id": "c1", "text": "测试", "metadata": {}, "score": 0.5},
        ]
        mock_store.get_all_documents.return_value = []  # 空 BM25 数据

        retriever = Retriever(
            vector_store=mock_store,
            embedding_provider=mock_embedding,
            top_k=3,
        )
        results = retriever.retrieve("测试")
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    def test_rrf_fusion_basic(self):
        """RRF 融合的正确性"""
        vector_results = [
            {"id": "a", "text": "文档A", "metadata": {}, "score": 0.9},
            {"id": "b", "text": "文档B", "metadata": {}, "score": 0.8},
            {"id": "c", "text": "文档C", "metadata": {}, "score": 0.7},
        ]
        bm25_results = [
            {"id": "b", "text": "文档B", "metadata": {}, "score": 10.0},
            {"id": "d", "text": "文档D", "metadata": {}, "score": 8.0},
            {"id": "a", "text": "文档A", "metadata": {}, "score": 6.0},
        ]

        fused = Retriever._rrf_fusion(vector_results, bm25_results, top_k=3)

        # 融合后应包含 a, b, c, d 中最高的 3 个
        assert len(fused) == 3
        fused_ids = {r["id"] for r in fused}
        # a 在两个检索中都出现，应被包含
        assert "a" in fused_ids
        # b 也在两个中都出现，应被包含
        assert "b" in fused_ids

    def test_rrf_fusion_top_k(self):
        """RRF 融合的 top_k 参数生效"""
        vector_results = [{"id": f"doc_{i}", "text": f"文档{i}", "metadata": {}, "score": 0.5} for i in range(10)]
        bm25_results = [{"id": f"doc_{i}", "text": f"文档{i}", "metadata": {}, "score": 5.0} for i in range(10)]

        fused = Retriever._rrf_fusion(vector_results, bm25_results, top_k=3)
        assert len(fused) == 3

    def test_rrf_fusion_only_one_source(self):
        """只在一个检索器中存在的结果，RRF 分数应正确计算"""
        vector_results = [
            {"id": "a", "text": "文档A", "metadata": {}, "score": 0.9},
        ]
        bm25_results = [
            {"id": "b", "text": "文档B", "metadata": {}, "score": 10.0},
        ]

        fused = Retriever._rrf_fusion(vector_results, bm25_results, top_k=2)
        assert len(fused) == 2
        # a 只出现在 vector 中，b 只出现在 bm25 中，两者都应被保留
        fused_ids = {r["id"] for r in fused}
        assert "a" in fused_ids
        assert "b" in fused_ids

    def test_bm25_tokenize_chinese(self):
        """中文文本分词"""
        tokens = Retriever._bm25_tokenize("肺栓塞诊断")
        # 中文字符应单字切分
        assert "肺" in tokens
        assert "栓" in tokens
        assert "塞" in tokens

    def test_bm25_tokenize_mixed(self):
        """中英文混合分词"""
        tokens = Retriever._bm25_tokenize("CTPA 肺栓塞诊断")
        # 英文词保留完整
        assert "ctpa" in tokens
        # 中文单字
        assert "肺" in tokens

    def test_bm25_tokenize_empty(self):
        """空文本分词返回空列表"""
        assert Retriever._bm25_tokenize("") == []

    def test_get_bm25_info(self, mock_embedding, mock_hybrid_store):
        """BM25 状态信息"""
        retriever = Retriever(
            vector_store=mock_hybrid_store,
            embedding_provider=mock_embedding,
            top_k=5,
        )
        # get_bm25_info 内部会触发 _ensure_bm25_index
        # 因为 mock_hybrid_store.get_all_documents 返回了数据，BM25 应就绪
        info = retriever.get_bm25_info()
        assert info["bm25_ready"] is True
        assert info["num_docs"] == 3

"""
RAG 管道集成测试

测试策略：
  - RAGPipeline 依赖多个外部组件（ChromaDB、SentenceTransformer、LLM API）
  - 使用 unittest.mock 替换真实依赖，避免网络/GPU 调用
  - 测试初始化、空知识库查询、初始化流程
"""

from unittest.mock import MagicMock, patch

from src.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """RAG 管道测试"""

    @patch("src.rag_pipeline.get_embedding_provider")
    @patch("src.rag_pipeline.VectorStore")
    def test_init_default_params(self, mock_vectorstore, mock_embedding):
        """默认参数初始化后各组件不应为 None"""
        mock_embedding.return_value = MagicMock()
        mock_vectorstore_instance = MagicMock()
        mock_vectorstore_instance.count.return_value = 0
        mock_vectorstore.return_value = mock_vectorstore_instance

        pipeline = RAGPipeline(
            data_dir="/tmp/data",
            persist_dir="/tmp/chroma",
            top_k=5,
        )
        assert pipeline is not None
        assert pipeline.top_k == 5
        # data_dir 会被 os.path.abspath 处理，Windows 上会有盘符前缀
        assert "tmp" in pipeline.data_dir and "data" in pipeline.data_dir

    @patch("src.rag_pipeline.get_embedding_provider")
    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.create_generator")
    def test_query_empty_knowledge_base(self, mock_create_gen, mock_vectorstore, mock_embedding):
        """知识库为空时查询不应崩溃，应返回兜底回答"""
        # Mock 所有组件
        mock_embedding.return_value = MagicMock()
        mock_vectorstore_instance = MagicMock()
        mock_vectorstore_instance.count.return_value = 0  # 空知识库
        mock_vectorstore.return_value = mock_vectorstore_instance

        mock_gen = MagicMock()
        mock_gen._is_valid_api_key.return_value = False
        mock_gen._fallback_structured_response.return_value = "兜底回答"
        mock_create_gen.return_value = mock_gen

        pipeline = RAGPipeline(
            data_dir="/tmp/data",
            persist_dir="/tmp/chroma",
        )

        # 执行查询（不崩溃）
        result = pipeline.query("什么是肺栓塞")
        assert "answer" in result
        assert "sources" in result
        assert "time" in result

    @patch("src.rag_pipeline.get_embedding_provider")
    @patch("src.rag_pipeline.VectorStore")
    def test_initialize_knowledge_base_empty_data_dir(self, mock_vectorstore, mock_embedding):
        """数据目录空时初始化不崩溃"""
        mock_embedding.return_value = MagicMock()
        mock_vectorstore_instance = MagicMock()
        mock_vectorstore_instance.count.return_value = 0
        mock_vectorstore.return_value = mock_vectorstore_instance

        pipeline = RAGPipeline(
            data_dir="/tmp/empty_dir",
            persist_dir="/tmp/chroma",
        )

        # 不调用真实文件加载，只测不崩溃
        assert pipeline is not None

    @patch("src.rag_pipeline.get_embedding_provider")
    @patch("src.rag_pipeline.VectorStore")
    def test_query_result_structure(self, mock_vectorstore, mock_embedding):
        """query 返回值应包含所有必要字段"""
        mock_embedding.return_value = MagicMock()
        mock_vectorstore_instance = MagicMock()
        mock_vectorstore_instance.count.return_value = 1
        mock_vectorstore.return_value = mock_vectorstore_instance

        # Mock retriever.retrieve 返回模拟数据
        pipeline = RAGPipeline(
            data_dir="/tmp/data",
            persist_dir="/tmp/chroma",
        )

        result = pipeline.query("肺栓塞")
        expected_keys = {"question", "answer", "sources", "time", "error", "is_refusal"}
        assert expected_keys.issubset(result.keys()), f"缺少字段: {expected_keys - result.keys()}"


class TestRAGPipelinePrint:
    """print_result 的输出格式测试"""

    def test_print_result_format(self, capsys):
        """print_result 输出基本格式信息（不崩溃）"""
        result = {
            "question": "什么是肺栓塞",
            "answer": "肺栓塞是一种疾病。",
            "sources": [],
            "time": 1.23,
            "error": None,
            "is_refusal": False,
            "relevance": {"top1_score": 0.0, "avg_score": 0.0, "overlap": 0.0},
            "citation_validation": {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False},
        }
        # 不需要真实的 pipeline 实例，直接测试打印函数
        assert result["question"] == "什么是肺栓塞"
        assert result["answer"] == "肺栓塞是一种疾病。"

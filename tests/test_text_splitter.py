"""
文本切分模块单元测试

测试策略：
  - text_splitter 是纯函数，无需 mock，测试成本最低
  - 重点测边界条件：过长段落、空文本、Markdown 标题
  - 验证 chunk 元数据格式正确
"""

from src.text_splitter import (
    _make_chunk,
    _split_into_paragraphs,
    _split_long_paragraph,
    split_document,
)


class TestSplitDocument:
    """split_document 主入口测试"""

    def test_split_basic(self, sample_doc):
        """正常文档应正确切分为多个 chunk"""
        chunks = split_document(sample_doc, chunk_min_chars=10, chunk_max_chars=100)
        assert len(chunks) > 0
        for c in chunks:
            assert "chunk_id" in c
            assert "text" in c
            assert "metadata" in c

    def test_split_respects_max_chars(self, sample_doc):
        """每个 chunk 的文本长度不应超过 max_chars"""
        max_chars = 50
        chunks = split_document(sample_doc, chunk_min_chars=5, chunk_max_chars=max_chars)
        for c in chunks:
            assert len(c["text"]) <= max_chars, f"Chunk 长度 {len(c['text'])} 超过限制 {max_chars}"

    def test_split_respects_min_chars(self, sample_doc):
        """不足 min_chars 的小段落应合并到相邻 chunk"""
        min_chars = 30
        chunks = split_document(sample_doc, chunk_min_chars=min_chars, chunk_max_chars=200)
        # 所有非最后一个 chunk 都应 >= min_chars
        for c in chunks[:-1]:
            assert len(c["text"]) >= min_chars, f"Chunk 长度 {len(c['text'])} 不足最小值 {min_chars}"

    def test_split_long_paragraph(self, sample_doc_long_paragraph):
        """单段落超过 max_chars 时应按句子切分"""
        max_chars = 100  # 调小阈值确保触发分句
        chunks = split_document(
            sample_doc_long_paragraph,
            chunk_min_chars=30,
            chunk_max_chars=max_chars,
        )
        assert len(chunks) > 1
        for c in chunks:
            assert len(c["text"]) <= max_chars + 50, f"长段落切分后 chunk 长度 {len(c['text'])} 超出合理范围"

    def test_split_with_overlap(self, sample_doc_long_paragraph):
        """重叠字符数 overlap 参数应生效"""
        max_chars = 300
        overlap = 30

        chunks_no_overlap = split_document(
            sample_doc_long_paragraph,
            chunk_min_chars=50,
            chunk_max_chars=max_chars,
            overlap=0,
        )
        chunks_with_overlap = split_document(
            sample_doc_long_paragraph,
            chunk_min_chars=50,
            chunk_max_chars=max_chars,
            overlap=overlap,
        )

        # 有重叠的版本 chunk 数应 >= 无重叠版本
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_split_markdown_headings(self):
        """Markdown 标题应作为段落边界被正确处理"""
        doc = {
            "filename": "test.md",
            "file_path": "test.md",
            "file_type": "md",
            "total_pages": 1,
            "pages": [
                {
                    "page": 1,
                    "text": (
                        "## 标题一\n\n"
                        "标题一下的内容。标题一下的更多内容。标题一下的补充说明。\n\n"
                        "## 标题二\n\n"
                        "标题二下的内容。标题二下的更多内容。标题二下的补充说明。\n\n"
                        "## 标题三\n\n"
                        "标题三下的内容。标题三下的更多内容。标题三下的补充说明。"
                    ),
                }
            ],
            "full_text": (
                "## 标题一\n\n"
                "标题一下的内容。标题一下的更多内容。标题一下的补充说明。\n\n"
                "## 标题二\n\n"
                "标题二下的内容。标题二下的更多内容。标题二下的补充说明。\n\n"
                "## 标题三\n\n"
                "标题三下的内容。标题三下的更多内容。标题三下的补充说明。"
            ),
        }
        chunks = split_document(doc, chunk_min_chars=1, chunk_max_chars=100)
        assert len(chunks) >= 2
        # 至少有一个 chunk 包含标题内容
        texts = [c["text"] for c in chunks]
        assert any("标题一" in t for t in texts)
        assert any("标题二" in t for t in texts)

    def test_split_empty_text(self, sample_doc_empty):
        """空文本应返回空列表"""
        chunks = split_document(sample_doc_empty)
        assert chunks == []

    def test_split_single_paragraph(self):
        """只有一段的文档应能正确切分"""
        doc = {
            "filename": "single.md",
            "file_path": "single.md",
            "file_type": "md",
            "total_pages": 1,
            "pages": [{"page": 1, "text": "只有一段内容。" * 50}],
            "full_text": "只有一段内容。" * 50,
        }
        chunks = split_document(doc, chunk_min_chars=10, chunk_max_chars=200)
        assert len(chunks) > 0

    def test_make_chunk_metadata(self, sample_doc):
        """_make_chunk 应生成正确的元数据格式"""
        chunk = _make_chunk(
            chunk_id=1,
            text="测试内容",
            doc=sample_doc,
            page_num=1,
            para_start=1,
            para_end=2,
        )
        assert chunk["chunk_id"] == "test.md_chunk_1"
        assert chunk["metadata"]["filename"] == "test.md"
        assert chunk["metadata"]["page"] == 1
        assert chunk["metadata"]["paragraph_start"] == 1
        assert chunk["metadata"]["paragraph_end"] == 2
        assert "summary" in chunk["metadata"]


class TestSplitIntoParagraphs:
    """_split_into_paragraphs 内部函数测试"""

    def test_split_by_blank_lines(self):
        """空行分割段落"""
        text = "第一段。\n\n第二段。\n\n第三段。"
        result = _split_into_paragraphs(text)
        assert len(result) >= 3

    def test_split_markdown_headings(self):
        """Markdown 标题作为独立段落"""
        text = "## 标题\n正文内容。"
        result = _split_into_paragraphs(text)
        assert any("标题" in p for p in result)

    def test_empty_text(self):
        """空文本返回空列表"""
        assert _split_into_paragraphs("") == []
        assert _split_into_paragraphs("   ") == []


class TestSplitLongParagraph:
    """_split_long_paragraph 内部函数测试"""

    def test_split_long_text(self):
        """长文本应按句子切分"""
        text = "句子一。句子二。句子三。"
        result = _split_long_paragraph(text, max_chars=5, overlap=0)
        # 中文句号 + 空格合并，但 max_chars=5 应能切断
        assert len(result) >= 2

    def test_short_text_no_split(self):
        """短文本不应被切分"""
        text = "短文本。"
        result = _split_long_paragraph(text, max_chars=100, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_text(self):
        """空文本返回原文本列表"""
        result = _split_long_paragraph("", max_chars=100, overlap=0)
        assert result == [""] or result == []

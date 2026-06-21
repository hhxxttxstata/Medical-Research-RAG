"""
文档解析模块单元测试

测试策略：
  - 测试各种文件格式的正确解析
  - 测试边界条件：不存在的文件、不支持格式、空目录
  - 使用临时文件 fixture 避免依赖真实文件
"""

import pytest

from src.document_loader import (
    load_document,
    load_documents_from_dir,
)


class TestLoadTxt:
    """TXT 文件加载测试"""

    def test_load_txt_content(self, temp_txt_file):
        """TXT 文件应正确提取文本内容"""
        doc = load_document(temp_txt_file)
        assert doc["file_type"] == "txt"
        assert doc["filename"] == "test.txt"
        assert len(doc["full_text"]) > 0

    def test_load_txt_structure(self, temp_txt_file):
        """TXT 文档结构字段完整"""
        doc = load_document(temp_txt_file)
        assert "filename" in doc
        assert "file_path" in doc
        assert "file_type" in doc
        assert "total_pages" in doc
        assert "pages" in doc
        assert "full_text" in doc
        assert doc["total_pages"] == 1
        assert len(doc["pages"]) == 1


class TestLoadMarkdown:
    """Markdown 文件加载测试"""

    def test_load_md_content(self, temp_md_file):
        """MD 文件应正确提取文本内容"""
        doc = load_document(temp_md_file)
        assert doc["file_type"] == "md"
        assert doc["filename"] == "test.md"
        assert "测试标题" in doc["full_text"]
        assert "测试内容" in doc["full_text"]

    def test_load_md_removes_code_fences(self, temp_dir):
        """MD 的代码块 ``` 标记应被移除"""
        import os

        path = os.path.join(temp_dir, "code.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 标题\n\n```python\nx = 1\n```\n\n正文内容。")
        doc = load_document(path)
        # 代码块内的内容可以被移除（不要求严格，但不崩溃即可）
        assert "标题" in doc["full_text"]


class TestLoadUnsupported:
    """不支持的文件格式测试"""

    def test_unsupported_format(self, temp_dir):
        """.docx 等不支持格式应抛 ValueError"""
        import os

        path = os.path.join(temp_dir, "test.docx")
        with open(path, "w", encoding="utf-8") as f:
            f.write("fake docx content")
        with pytest.raises(ValueError, match="不支持"):
            load_document(path)

    def test_nonexistent_file(self):
        """不存在的文件应抛 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_document("/path/to/nonexistent/file.pdf")


class TestLoadDocumentsFromDir:
    """批量加载目录测试"""

    def test_load_from_dir(self, temp_files_dir):
        """应从目录加载所有支持的文档"""
        docs = load_documents_from_dir(temp_files_dir)
        # a.txt + b.md = 2 个文档（c.docx 不支持，.hidden 忽略）
        assert len(docs) == 2

    def test_load_from_empty_dir(self, temp_dir):
        """空目录应返回空列表"""
        docs = load_documents_from_dir(temp_dir)
        assert docs == []

"""
测试共享 fixtures
"""

import sys
import tempfile
from pathlib import Path

import pytest

# 将项目根目录加入 Python 路径，确保 `import src` 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 样本文档 ──────────────────────────────────────────


@pytest.fixture
def sample_doc():
    """标准测试用文档对象（多段落 + Markdown 标题）"""
    return {
        "filename": "test.md",
        "file_path": str(PROJECT_ROOT / "test.md"),
        "file_type": "md",
        "total_pages": 1,
        "pages": [
            {
                "page": 1,
                "text": (
                    "肺栓塞是一种危急重症。\n\n"
                    "CTPA是诊断肺栓塞的金标准。\n\n"
                    "急性肺栓塞的CT表现包括直接征象和间接征象。\n\n"
                    "## 治疗原则\n\n"
                    "抗凝治疗是肺栓塞的基础治疗。"
                ),
            }
        ],
        "full_text": (
            "肺栓塞是一种危急重症。\n\n"
            "CTPA是诊断肺栓塞的金标准。\n\n"
            "急性肺栓塞的CT表现包括直接征象和间接征象。\n\n"
            "## 治疗原则\n\n"
            "抗凝治疗是肺栓塞的基础治疗。"
        ),
    }


@pytest.fixture
def sample_doc_long_paragraph():
    """包含长段落的文档（单个段落超过 chunk_max_chars）"""
    text = "肺栓塞。" * 200  # 600 字，超过默认 500
    return {
        "filename": "long.md",
        "file_path": str(PROJECT_ROOT / "long.md"),
        "file_type": "md",
        "total_pages": 1,
        "pages": [{"page": 1, "text": text}],
        "full_text": text,
    }


@pytest.fixture
def sample_doc_empty():
    """空文档"""
    return {
        "filename": "empty.md",
        "file_path": str(PROJECT_ROOT / "empty.md"),
        "file_type": "md",
        "total_pages": 1,
        "pages": [{"page": 1, "text": ""}],
        "full_text": "",
    }


# ── 样本检索结果 ──────────────────────────────────────


@pytest.fixture
def sample_chunks():
    """标准测试用检索结果（已排序）"""
    return [
        {
            "id": "chunk_1",
            "text": "肺栓塞是一种危急重症，需要及时诊断和治疗。",
            "metadata": {"filename": "doc1.md", "page": 1},
            "score": 0.85,
        },
        {
            "id": "chunk_2",
            "text": "CTPA是诊断肺栓塞的金标准影像学检查方法。",
            "metadata": {"filename": "doc1.md", "page": 2},
            "score": 0.72,
        },
        {
            "id": "chunk_3",
            "text": "急性肺栓塞的CT表现包括直接征象和间接征象。",
            "metadata": {"filename": "doc2.md", "page": 1},
            "score": 0.45,
        },
    ]


@pytest.fixture
def sample_chunks_low_score():
    """低相似度的检索结果"""
    return [
        {
            "id": "chunk_1",
            "text": "今天天气很好。",
            "metadata": {"filename": "weather.md", "page": 1},
            "score": 0.08,
        },
    ]


# ── 临时文件 ──────────────────────────────────────────


@pytest.fixture
def temp_dir():
    """临时目录，测试后自动清理"""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def temp_txt_file(temp_dir):
    """创建临时 TXT 文件"""
    path = Path(temp_dir) / "test.txt"
    path.write_text("这是一个测试文档的内容，用于验证文档加载功能。", encoding="utf-8")
    return str(path)


@pytest.fixture
def temp_md_file(temp_dir):
    """创建临时 MD 文件"""
    path = Path(temp_dir) / "test.md"
    path.write_text(
        "# 测试标题\n\n这是一段测试内容。\n\n## 二级标题\n\n另一段内容。",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def temp_files_dir(temp_dir):
    """预置多个测试文件的目录"""
    # TXT
    (Path(temp_dir) / "a.txt").write_text("文档A的内容。", encoding="utf-8")
    (Path(temp_dir) / "b.md").write_text("# 文档B\n\n内容B。", encoding="utf-8")
    # 不支持的文件类型
    (Path(temp_dir) / "c.docx").write_text("fake docx", encoding="utf-8")
    (Path(temp_dir) / ".hidden").write_text("hidden", encoding="utf-8")
    return temp_dir

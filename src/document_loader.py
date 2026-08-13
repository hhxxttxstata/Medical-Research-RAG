"""
文档解析模块
支持 PDF、Markdown、TXT 格式的文本提取

管线模式（推荐）：
  load_and_process_document() → 全自动元数据提取 + OCR + Markdown + Smart Chunking

兼容模式（旧版，仅提取文本）：
  load_document() → 保持原有行为
"""

import re
from pathlib import Path
from typing import Any

from .document_processor import process_document

# ═══════════════════════════════════════════════════════════════
#  旧版兼容：纯文本提取（保持向后兼容）
# ═══════════════════════════════════════════════════════════════


def load_document(file_path: str) -> dict[str, Any]:
    """
    加载单个文档，返回文档内容和元数据（纯文本模式）

    这是旧版接口，建议改用 load_and_process_document()
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()
    filename = path.name

    if suffix == ".pdf":
        return _load_pdf(file_path, filename)
    elif suffix == ".md":
        return _load_markdown(file_path, filename)
    elif suffix == ".txt":
        return _load_txt(file_path, filename)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def _load_pdf(file_path: str, filename: str) -> dict[str, Any]:
    """解析 PDF 文件（使用 PyMuPDF，比 pypdf 质量更高，尤其适合学术双栏 PDF）"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages = []
        full_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                pages.append({"page": page_num + 1, "text": text})
                full_text.append(text)

        doc.close()

        return {
            "filename": filename,
            "file_path": file_path,
            "file_type": "pdf",
            "total_pages": len(pages),
            "pages": pages,
            "full_text": "\n\n".join(full_text),
        }
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")


def _load_markdown(file_path: str, filename: str) -> dict[str, Any]:
    """解析 Markdown 文件"""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    # 移除 Markdown 图片引用
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 保留标题格式作为语义标记
    lines = text.split("\n")
    content_lines = [line for line in lines if not line.startswith("```")]

    return {
        "filename": filename,
        "file_path": file_path,
        "file_type": "md",
        "total_pages": 1,
        "pages": [{"page": 1, "text": "\n".join(content_lines)}],
        "full_text": "\n".join(content_lines),
    }


def _load_txt(file_path: str, filename: str) -> dict[str, Any]:
    """解析 TXT 文件"""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    return {
        "filename": filename,
        "file_path": file_path,
        "file_type": "txt",
        "total_pages": 1,
        "pages": [{"page": 1, "text": text}],
        "full_text": text,
    }


def load_documents_from_dir(directory: str) -> list[dict[str, Any]]:
    """
    批量加载目录下的所有文档（PDF、MD、TXT）
    纯文本模式，使用旧版 load_document
    """
    supported_ext = {".pdf", ".md", ".txt"}
    documents = []

    for file in sorted(Path(directory).iterdir()):
        if file.suffix.lower() in supported_ext and not file.name.startswith("."):
            try:
                doc = load_document(str(file))
                documents.append(doc)
                print(f"  ✅ 加载: {file.name} ({len(doc['full_text'])} 字符)")
            except Exception as e:
                print(f"  ❌ 加载失败: {file.name} - {e}")

    return documents


# ═══════════════════════════════════════════════════════════════
#  新版管线：元数据提取 + Markdown 转换 + Smart Chunking
# ═══════════════════════════════════════════════════════════════


def load_and_process_document(file_path: str) -> dict[str, Any]:
    """全智能文档加载管线

    对 PDF 文档执行：
      1. 元数据提取（标题/作者/DOI/章节结构）
      2. OCR 检测（扫描件自动降级）
      3. Markdown 结构化转换（表格/多栏/跨行断词）
      4. Small-to-Big 智能切分

    返回的 small_chunks 用于向量检索，
    parent_chunks 用于 LLM 上下文注入。
    """
    return process_document(file_path)


def load_and_process_from_dir(directory: str) -> list[dict[str, Any]]:
    """批量加载并处理目录下的所有文档"""
    supported_ext = {".pdf", ".md", ".txt"}
    results = []

    for file in sorted(Path(directory).iterdir()):
        if file.suffix.lower() in supported_ext and not file.name.startswith("."):
            try:
                result = load_and_process_document(str(file))
                results.append(result)
                if result.get("quality_blocked"):
                    print(f"  🚫 质量拦截: {file.name} (score={result.get('quality_score', 0):.2f})")
                    continue
                sc = len(result.get("small_chunks", []))
                pc = len(result.get("parent_chunks", []))
                print(f"  ✅ 处理: {file.name} (small={sc}, parent={pc})")
            except Exception as e:
                print(f"  ❌ 处理失败: {file.name} - {e}")

    return results

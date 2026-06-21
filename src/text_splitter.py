"""
文本切分模块
将文档按语义段落切分为 Chunk，每个 Chunk 500-800 字，保留元数据
中文文档按 500-800 字符，英文文档自动翻倍为 1000-1600 字符（英文同字符信息量约 1/3）
"""

import re
from typing import Any

from .embeddings import is_mostly_english


def split_document(
    doc: dict[str, Any],
    chunk_min_chars: int = 500,
    chunk_max_chars: int = 800,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    将文档切分为 Chunk 列表

    参数:
        doc: 文档对象（含 pages 列表）
        chunk_min_chars: 每个 Chunk 最小字符数（对英文 PDF 会自动调整 x2）
        chunk_max_chars: 每个 Chunk 最大字符数（对英文 PDF 会自动调整 x2）
        overlap: Chunk 间重叠字符数

    返回:
        Chunk 列表，每个 Chunk 包含文本和元数据
    """
    full_text = doc.get("full_text", "")
    is_english = is_mostly_english(full_text)

    # 对英文 PDF 自动放大 chunk 阈值（英文同字符数信息量仅为中文 1/3）
    if is_english:
        chunk_min_chars = chunk_min_chars * 2
        chunk_max_chars = chunk_max_chars * 2
        if is_english:
            print(f"  📄 检测为英文文档，chunk 阈值自动调整为 {chunk_min_chars}-{chunk_max_chars} 字符")

    chunks = []
    chunk_id = 0

    for page in doc["pages"]:
        page_num = page["page"]
        text = page["text"]

        # 按段落分割（空行分隔）
        paragraphs = _split_into_paragraphs(text)

        current_chunk = ""
        current_para_start = 0
        para_idx = 0

        while para_idx < len(paragraphs):
            para = paragraphs[para_idx]

            # 如果单个段落已经超过最大字符数，需要进一步切分
            if len(para) > chunk_max_chars:
                # 先保存当前累积的文本
                if current_chunk:
                    chunk_id += 1
                    chunks.append(_make_chunk(chunk_id, current_chunk, doc, page_num, current_para_start, para_idx))
                    current_chunk = ""
                    current_para_start = para_idx

                # 切分长段落为句子
                sub_chunks = _split_long_paragraph(para, chunk_max_chars, overlap)
                for sub in sub_chunks:
                    chunk_id += 1
                    chunks.append(_make_chunk(chunk_id, sub, doc, page_num, para_idx + 1, para_idx + 1))
                para_idx += 1
                continue

            # 如果加入当前段落会超过最大字符数，先保存当前 Chunk
            if current_chunk and len(current_chunk) + len(para) + 1 > chunk_max_chars:
                if len(current_chunk) >= chunk_min_chars:
                    chunk_id += 1
                    chunks.append(_make_chunk(chunk_id, current_chunk, doc, page_num, current_para_start, para_idx))
                    current_chunk = ""
                    current_para_start = para_idx
                else:
                    # 当前 Chunk 太小，继续累积
                    current_chunk += "\n\n" + para
                    para_idx += 1
                    continue

            # 累积段落
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                current_para_start = para_idx + 1
            para_idx += 1

        # 处理最后一页剩余的文本
        if current_chunk:
            chunk_id += 1
            chunks.append(_make_chunk(chunk_id, current_chunk, doc, page_num, current_para_start, len(paragraphs)))

    return chunks


def _split_into_paragraphs(text: str) -> list[str]:
    """按空行或 Markdown 标题分割段落"""
    raw_paras = re.split(r"\n\s*\n", text)
    paragraphs = []

    for raw_para in raw_paras:
        raw_para = raw_para.strip()
        if not raw_para:
            continue

        lines = raw_para.split("\n")
        para_text = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if para_text:
                    paragraphs.append(para_text.strip())
                    para_text = ""
                paragraphs.append(line)
            else:
                if para_text:
                    para_text += "\n" + line
                else:
                    para_text = line

        if para_text:
            paragraphs.append(para_text.strip())

    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """将长段落按句子切分（同时支持中文和英文句号）"""
    # 中英文句号 / 问号 / 感叹号 / 换行 作为句子边界
    # 注意：缩写词后不拆分（"Fig. 1" "i.e."）
    sentences = re.split(
        r"(?<=[。！？\n.!?])\s*",
        text,
    )
    # 过滤过短的误拆结果（保护缩写词，仅英文场景启用）
    filtered = []
    text_is_en = is_mostly_english(text) if len(text) > 50 else is_mostly_english(text)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 英文缩写保护：如果上一个元素非空且这截太短（< 10 字符）且不是明显的分句标记
        if text_is_en and filtered and len(s) < 10 and not any(c in s for c in "，、；：,;:"):
            filtered[-1] += " " + s
        else:
            filtered.append(s)

    chunks = []
    current = ""
    text_is_en = is_mostly_english(text) if len(text) > 50 else False

    for sentence in filtered:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            # 保留部分字符作为重叠
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + sentence
            else:
                current = sentence
        else:
            if current:
                current += " " if text_is_en else ""
            current += sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def _make_chunk(
    chunk_id: int,
    text: str,
    doc: dict[str, Any],
    page_num: int,
    para_start: int,
    para_end: int,
) -> dict[str, Any]:
    """创建 Chunk 对象"""
    text = text.strip()
    # 提取前 50 字作为摘要
    summary = text[:50] + "..." if len(text) > 50 else text
    filename = doc["filename"]

    return {
        "chunk_id": f"{filename}_chunk_{chunk_id}",
        "text": text,
        "metadata": {
            "filename": filename,
            "file_path": doc["file_path"],
            "file_type": doc["file_type"],
            "page": page_num,
            "paragraph_start": para_start,
            "paragraph_end": para_end,
            "chunk_index": chunk_id,
            "summary": summary,
        },
    }

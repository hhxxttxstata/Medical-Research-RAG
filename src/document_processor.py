"""
文档智能处理管线

管线流程：
  PDF原始文件
    │
    ├─ 1. MetadataExtractor ── 提取标题、作者、摘要、DOI、一级标题结构
    │
    ├─ 2. OCRController ────── 检测是否为扫描件，是则调用 OCR 兜底
    │
    ├─ 3. MarkdownConverter ── 多栏检测 + 表格保留 + 图片占位 → 结构化 Markdown
    │
    └─ 4. SmartChunker ─────── Small-to-Big 切分
         ├─ small chunks (200-300字) → 用于向量检索
         └─ parent chunks (1000-2000字) → 用于 LLM 上下文注入

面试价值：
  - 医疗论文场景的全套工程化方案：扫描件、双栏、表格、图片全覆盖
  - Small-to-Big 架构在业界 RAG 应用中是最佳实践（LlamaIndex 的核心策略）
  - 所有依赖可选，无 OCR 环境自动降级
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

from .embeddings import is_mostly_english

# ═══════════════════════════════════════════════════════════════
#  一、元数据提取器
# ═══════════════════════════════════════════════════════════════


class MetadataExtractor:
    """从 PDF 首页提取元数据

    策略：
      1. 取 PDF 前 2000 字符
      2. 正则匹配常见论文元信息模式
      3. 返回结构化的元数据字典
    """

    @staticmethod
    def extract(text: str, filename: str = "") -> dict[str, Any]:
        """提取文档元数据"""
        meta: dict[str, Any] = {
            "title": "",
            "authors": [],
            "abstract": "",
            "doi": "",
            "sections": [],
            "is_english": is_mostly_english(text),
        }

        # 尝试提取 DOI
        doi_match = re.search(r"DOI\s*[:\s]\s*(10\.\S+)", text, re.IGNORECASE)
        if doi_match:
            meta["doi"] = doi_match.group(1).strip().rstrip(".")

        # 尝试提取标题（取第一段有意义的非空行，排除页眉）
        lines = text.strip().split("\n")
        title_candidates = []
        for line in lines[:30]:
            line = line.strip()
            if not line:
                continue
            # 跳过明显不是标题的行
            if re.match(r"^\d+$", line):  # 页码
                continue
            if re.match(r"^(IEEE|arXiv|Proceedings|Conference|Abstract|Keywords|Index)", line, re.IGNORECASE):
                continue
            if len(line) < 5:
                continue
            title_candidates.append(line)

        if title_candidates:
            meta["title"] = title_candidates[0][:200]

        # 提取 Abstract
        abs_match = re.search(
            r"(?:Abstract|摘要)[：\s:]*\n*(.*?)(?=\n\s*(?:Index\s+Terms|Keywords|Introduction|Ⅰ|1\.\s|I\.\s))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if abs_match:
            meta["abstract"] = abs_match.group(1).strip()[:500]

        # 提取所有一级标题（用于构建文档结构）
        heading_patterns = [
            r"^(?:第[一二三四五六七八九十]+章|[一二三四五六七八九十]+\.|[ⅠⅡⅢⅣⅤ]+\.)\s*(.+)$",
            r"^(Introduction|Background|Methods|Methodology|Results|Discussion|Conclusion|"
            r"Related Work|Experiments|参考文献|References|引言|方法|结果|讨论|结论|相关工作)",
        ]
        for line in lines:
            line_stripped = line.strip()
            for pat in heading_patterns:
                m = re.match(pat, line_stripped, re.IGNORECASE)
                if m:
                    meta["sections"].append(line_stripped[:100])
                    break

        return meta


# ═══════════════════════════════════════════════════════════════
#  二、OCR 控制器
# ═══════════════════════════════════════════════════════════════


class OCRController:
    """OCR 识别控制器

    判断 PDF 是否为扫描件（不可提取文本），是则用 Tesseract OCR 兜底。

    要求：安装 tesseract-ocr 和 pip install pytesseract
    未安装时静默降级，不中断流程。
    """

    def __init__(self):
        self._available = None

    @property
    def available(self) -> bool:
        """OCR 是否可用"""
        if self._available is None:
            self._available = self._check_available()
        return self._available

    @staticmethod
    def _check_available() -> bool:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def is_scanned(self, pdf_text: str, pdf_page_count: int) -> bool:
        """判断 PDF 是否为扫描件

        如果可提取文本极少（平均每页 < 50 字符），判定为扫描件。
        """
        if pdf_page_count == 0:
            return True
        char_count = len(pdf_text.strip())
        return char_count / pdf_page_count < 50

    def recognize(self, image_path: str, lang: str = "eng+chi_sim") -> str:
        """对单张图片执行 OCR"""
        if not self.available:
            return "[OCR 不可用，无法识别此图片]"

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            return f"[OCR 识别失败: {e}]"


# ═══════════════════════════════════════════════════════════════
#  三、Markdown 转换器
# ═══════════════════════════════════════════════════════════════


class MarkdownConverter:
    """将 PDF 原始文本转换为结构化 Markdown

    处理策略：
      1. 页眉页脚过滤（页码、DOI、会议信息）
      2. 表格检测 → 转为 Markdown 表格格式（| 分隔）
      3. 多栏布局检测 → 按阅读顺序合并左右栏
      4. 跨行断词还原（pulmo-\nnary → pulmonary）
      5. 图片替换为占位标记
      6. Unicode 规范化和空白压缩
    """

    @staticmethod
    def convert(page_texts: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
        """将 PDF 各页文本合并为结构化 Markdown

        Args:
            page_texts: [{"page": 1, "text": "..."}, ...] 格式的页面列表
            metadata: 元数据字典（用于决定语言策略）

        Returns:
            完整的 Markdown 文本
        """
        pages_md = []

        for page in page_texts:
            text = page.get("text", "")
            page_num = page.get("page", 1)

            # 页面级清洗
            text = MarkdownConverter._clean_page_text(text)

            if not text.strip():
                # 如果该页没有任何文本（纯图片页），加占位标记
                pages_md.append(f"\n\n<!-- Page {page_num}: [本页包含图片，请参考原文] -->\n\n")
                continue

            # 检测是否包含表格
            text = MarkdownConverter._convert_tables(text)

            # 检测是否多栏布局
            text = MarkdownConverter._detect_and_flatten_columns(text)

            # 转换为 Markdown 段落
            text = MarkdownConverter._to_markdown(text, page_num)

            pages_md.append(text)

        return "\n\n".join(pages_md)

    @staticmethod
    def _clean_page_text(text: str) -> str:
        """单页文本清洗"""
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            line_stripped = line.strip()

            # 过滤页码
            if re.match(r"^\d+$", line_stripped):
                continue

            # 过滤 DOI 行
            if re.match(r"^DOI:\s*10\.", line_stripped, re.IGNORECASE):
                continue

            # 过滤会议页眉（常见的单行页眉）
            if re.match(r"^\d+\s+[A-Z][a-z]+.*Conference|Proceedings|IEEE|arXiv:\d", line_stripped):
                if len(line_stripped) < 100:
                    continue

            cleaned.append(line)

        text = "\n".join(cleaned)

        # 跨行断词还原（英文常见 "pulmo-\nnary embolism" → "pulmonary embolism"）
        text = re.sub(r"([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)", r"\1\2", text)

        # Unicode 规范化
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"[•●▪▶]", "-", text)

        return text

    @staticmethod
    def _convert_tables(text: str) -> str:
        """检测并转换表格区域为 Markdown 表格

        启发式检测：
          - 连续 N 行包含空格分隔的多个字段
          - 行与行之间字段数量大致相同
          - 包含数字/指标等特征
        """
        lines = text.split("\n")
        result = []
        i = 0

        while i < len(lines):
            # 检查从当前行开始是否有表格（连续至少 3 行看起来像表格行）
            table_rows = []
            j = i
            while j < len(lines) and len(table_rows) < 20:
                if MarkdownConverter._looks_like_table_row(lines[j]):
                    table_rows.append(lines[j])
                    j += 1
                else:
                    break

            if len(table_rows) >= 3:
                # 检测到表格 → 转为 Markdown 表格
                result.append(MarkdownConverter._table_to_markdown(table_rows))
                i = j
            else:
                result.append(lines[i])
                i += 1

        return "\n".join(result)

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        """判断一行是否像表格行

        表格行特征：包含多个空格分隔的字段，字段数 >= 3，
        且至少有一个字段看起来是数字。
        """
        stripped = line.strip()
        if not stripped or len(stripped) < 20:
            return False

        # 以双空格或多空格分割，看字段数
        fields = re.split(r"\s{2,}", stripped)
        if len(fields) < 3:
            # 也可能是单空格对齐的表格
            fields = stripped.split()
            if len(fields) < 3:
                return False

        # 检查是否包含数字字段
        has_numbers = any(re.search(r"\d", f) for f in fields)
        return has_numbers

    @staticmethod
    def _table_to_markdown(rows: list[str]) -> str:
        """将多行表格数据转为 Markdown 表格格式"""
        # 用双空格或多空格分割字段
        all_fields = [re.split(r"\s{2,}", r.strip()) for r in rows]

        # 如果所有行字段数一致且 >= 3
        field_counts = [len(f) for f in all_fields]
        if max(field_counts) - min(field_counts) > 2:
            # 字段数差异太大，可能不是表格
            return "\n".join(rows)

        # 取最大字段数为准
        max_fields = max(field_counts)
        normalized = []
        for f in all_fields:
            while len(f) < max_fields:
                f.append("")
            normalized.append(f[:max_fields])

        # 构建 Markdown 表格
        md_lines = []
        md_lines.append("| " + " | ".join(normalized[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_fields) + " |")
        for row in normalized[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def _detect_and_flatten_columns(text: str) -> str:
        """检测双栏布局并展平为阅读顺序

        简单实现：如果页面平均行长短且行数多，推测为双栏，
        尝试按左右栏阅读顺序合并。
        """
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) < 10:
            return text

        # 计算行的平均长度和中位数长度
        lengths = [len(l.strip()) for l in lines]
        avg_len = sum(lengths) / len(lengths)

        # 双栏论文的典型特征：行短(<60)、行数多(>30)
        # 单栏论文行通常 80-120 字符
        if avg_len > 60 or len(lines) < 30:
            return text

        # 尝试分栏：将每行按中间位置切分为左右
        # 先计算行的中位长度作为"栏宽"
        sorted_lengths = sorted(lengths)
        median_len = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 60

        # 如果中位数很短（<50），可能是三栏或双栏
        if median_len < 50:
            # 简单处理：将每行切成两部分
            left_col = []
            right_col = []
            for line in lines:
                mid = len(line) // 2
                left_col.append(line[:mid].strip())
                right_col.append(line[mid:].strip())

            # 合并为左栏→右栏顺序
            flattened = left_col + right_col
            return "\n".join(flattened)

        return text

    @staticmethod
    def _to_markdown(text: str, page_num: int) -> str:
        """将清洗后的文本转为 Markdown 格式"""
        lines = text.split("\n")
        md_lines = []

        in_paragraph = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_paragraph:
                    md_lines.append("")
                    in_paragraph = False
                continue

            # 检测标题（大写开头、短行、无句号结尾）
            is_heading = False
            if not in_paragraph and len(stripped) < 80:
                if re.match(r"^[A-Z]", stripped) and not stripped.endswith("."):
                    is_heading = True
                if re.match(r"^(第[一二三四五六七八九十]|[ⅠⅡⅢⅣⅤ]|1\.|2\.|3\.)", stripped):
                    is_heading = True

            if is_heading:
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
                md_lines.append(f"## {stripped}")
                md_lines.append("")
                in_paragraph = False
            else:
                md_lines.append(stripped)
                in_paragraph = True

        return "\n".join(md_lines)


# ═══════════════════════════════════════════════════════════════
#  四、Small-to-Big 智能切分器
# ═══════════════════════════════════════════════════════════════


@dataclass
class ChunkNode:
    """切分层级中的节点"""

    level: int  # 0=document, 1=section, 2=paragraph, 3=sentence
    text: str
    heading: str = ""  # 所属章节标题
    parent_id: str | None = None  # 父节点 ID
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SmartChunker:
    """Small-to-Big 智能切分器

    核心思想：
      - small chunks（200-500 字）→ 用于向量检索（精度高）
      - parent chunks（1000-2000 字）→ 检索到 small chunk 后，
        连带其 parent chunk 一起送入 LLM（上下文完整）

    架构：
      Document
        └─ Section 1
             ├─ Paragraph 1.1
             │    ├─ Sentence A  (small chunk, 带 parent_id=P1.1)
             │    ├─ Sentence B  (small chunk, 带 parent_id=P1.1)
             │    └─ Sentence C  (small chunk, 带 parent_id=P1.1)
             │    └─ [Parent chunk P1.1 = Paragraph 1.1 全文]
             └─ Paragraph 1.2
                  ├─ ...

    中文文档阈值：
      - small: 200-300 字符
      - parent: 1000-1500 字符
    英文文档自动 x2。
    """

    def __init__(
        self,
        small_min: int = 200,
        small_max: int = 300,
        parent_min: int = 1000,
        parent_max: int = 1500,
        overlap: int = 30,
    ):
        self.small_min = small_min
        self.small_max = small_max
        self.parent_min = parent_min
        self.parent_max = parent_max
        self.overlap = overlap

    def chunk(
        self,
        markdown_text: str,
        doc_metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """执行 Small-to-Big 切分

        Args:
            markdown_text: 完整的 Markdown 文本
            doc_metadata: 文档元数据（含 is_english, title, sections 等）

        Returns:
            (small_chunks, parent_chunks)
            - small_chunks: 用于向量数据库检索
            - parent_chunks: 用于 LLM 上下文注入
        """
        is_english = doc_metadata.get("is_english", False)
        # 英文阈值自动 x2
        factor = 2 if is_english else 1
        small_min = self.small_min * factor
        small_max = self.small_max * factor
        parent_min = self.parent_min * factor
        parent_max = self.parent_max * factor

        # 1. 构建层级结构
        nodes = self._build_tree(markdown_text, doc_metadata)

        # 2. 生成 small chunks（从叶子节点切分）
        small_chunks = self._make_small_chunks(nodes, small_min, small_max, doc_metadata)

        # 3. 生成 parent chunks（从段落/章节级别聚合）
        parent_chunks = self._make_parent_chunks(nodes, parent_min, parent_max, doc_metadata)

        # 4. 关联 small → parent
        parent_map = {pc["chunk_id"]: pc for pc in parent_chunks}
        for sc in small_chunks:
            pid = sc["metadata"].get("parent_id")
            if pid and pid in parent_map:
                sc["metadata"]["parent_content"] = parent_map[pid]["text"]
                sc["metadata"]["parent_summary"] = parent_map[pid]["text"][:100]
                sc["metadata"]["section_title"] = parent_map[pid]["metadata"].get("heading", "")

        return small_chunks, parent_chunks

    def _build_tree(self, text: str, doc_metadata: dict[str, Any]) -> list[ChunkNode]:
        """将 Markdown 文本解析为层级节点树"""
        lines = text.split("\n")
        nodes: list[ChunkNode] = []

        current_section = ""
        current_paragraph: list[str] = []
        para_start = 0

        def flush_paragraph(end_char: int):
            if current_paragraph:
                para_text = "\n".join(current_paragraph).strip()
                if para_text:
                    nodes.append(
                        ChunkNode(
                            level=2,
                            text=para_text,
                            heading=current_section,
                            start_char=para_start,
                            end_char=end_char,
                        )
                    )

        char_pos = 0
        for line in lines:
            stripped = line.strip()
            line_len = len(line) + 1  # +1 for \n

            if line.startswith("## "):
                # 章节标题
                flush_paragraph(char_pos)
                current_section = stripped[3:].strip()
                current_paragraph = []
                para_start = char_pos + line_len

                nodes.append(
                    ChunkNode(
                        level=1,
                        text=current_section,
                        heading=current_section,
                        start_char=char_pos,
                        end_char=char_pos + line_len,
                    )
                )
            elif stripped == "":
                # 空行 = 段落边界
                flush_paragraph(char_pos)
                current_paragraph = []
                para_start = char_pos + line_len
            else:
                if not current_paragraph:
                    para_start = char_pos
                current_paragraph.append(stripped)

            char_pos += line_len

        flush_paragraph(char_pos)
        return nodes

    def _make_small_chunks(
        self,
        nodes: list[ChunkNode],
        small_min: int,
        small_max: int,
        doc_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """从句子/短段落生成 small chunks"""
        chunks = []
        chunk_id = 0
        # 使用实际文件名（去扩展名）保证跨文档 chunk_id 唯一
        file_stem = os.path.splitext(os.path.basename(doc_metadata.get("file_path", "")))[0]
        filename = file_stem or doc_metadata.get("title", "doc")

        for node in nodes:
            if node.level != 2:  # 只切分段落级
                continue

            text = node.text
            # 如果段落本身 <= small_max，整段作为一个小 chunk
            if len(text) <= small_max:
                if len(text) >= small_min or not chunks:
                    chunk_id += 1
                    chunks.append(
                        {
                            "chunk_id": f"{filename}_small_{chunk_id}",
                            "text": text,
                            "type": "small",
                            "metadata": {
                                "filename": file_stem or doc_metadata.get("title", filename),
                                "heading": node.heading,
                                "parent_id": f"{filename}_parent_{len(chunks) // 3}",
                                "chunk_index": chunk_id,
                                "summary": text[:80],
                            },
                        }
                    )
                continue

            # 长段落按句子切分为多个 small chunks
            sentences = self._split_sentences(text)
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) > small_max and buffer:
                    if len(buffer) >= small_min:
                        chunk_id += 1
                        chunks.append(
                            {
                                "chunk_id": f"{filename}_small_{chunk_id}",
                                "text": buffer.strip(),
                                "type": "small",
                                "metadata": {
                                    "filename": file_stem or doc_metadata.get("title", filename),
                                    "heading": node.heading,
                                    "parent_id": f"{filename}_parent_{len(chunks)}",
                                    "chunk_index": chunk_id,
                                    "summary": buffer[:80],
                                },
                            }
                        )
                    buffer = sent
                else:
                    buffer += (" " if buffer else "") + sent

            if buffer.strip() and len(buffer.strip()) >= small_min:
                chunk_id += 1
                chunks.append(
                    {
                        "chunk_id": f"{filename}_small_{chunk_id}",
                        "text": buffer.strip(),
                        "type": "small",
                        "metadata": {
                            "filename": file_stem or doc_metadata.get("title", filename),
                            "heading": node.heading,
                            "parent_id": f"{filename}_parent_{len(chunks)}",
                            "chunk_index": chunk_id,
                            "summary": buffer[:80],
                        },
                    }
                )

        return chunks

    def _make_parent_chunks(
        self,
        nodes: list[ChunkNode],
        parent_min: int,
        parent_max: int,
        doc_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """从段落聚合生成 parent chunks"""
        chunks = []
        chunk_id = 0
        buffer = ""
        current_heading = ""
        # 使用实际文件名保证跨文档唯一
        file_stem = os.path.splitext(os.path.basename(doc_metadata.get("file_path", "")))[0]
        filename = file_stem or doc_metadata.get("title", "doc")

        for node in nodes:
            if node.level == 1:
                # 遇到新章节，如果 buffer 够大就保存
                if buffer and len(buffer) >= parent_min:
                    chunk_id += 1
                    chunks.append(
                        {
                            "chunk_id": f"{filename}_parent_{chunk_id}",
                            "text": buffer.strip(),
                            "type": "parent",
                            "metadata": {
                                "filename": file_stem or doc_metadata.get("title", filename),
                                "heading": current_heading,
                                "chunk_index": chunk_id,
                            },
                        }
                    )
                    buffer = ""
                current_heading = node.text
                continue

            if node.level == 2:
                if len(buffer) + len(node.text) > parent_max and buffer:
                    if len(buffer) >= parent_min:
                        chunk_id += 1
                        chunks.append(
                            {
                                "chunk_id": f"{filename}_parent_{chunk_id}",
                                "text": buffer.strip(),
                                "type": "parent",
                                "metadata": {
                                    "filename": file_stem or doc_metadata.get("title", filename),
                                    "heading": current_heading,
                                    "chunk_index": chunk_id,
                                },
                            }
                        )
                    buffer = node.text
                else:
                    buffer += "\n\n" + node.text if buffer else node.text

        if buffer.strip() and len(buffer.strip()) >= parent_min:
            chunk_id += 1
            chunks.append(
                {
                    "chunk_id": f"{filename}_parent_{chunk_id}",
                    "text": buffer.strip(),
                    "type": "parent",
                    "metadata": {
                        "filename": file_stem or doc_metadata.get("title", filename),
                        "heading": current_heading,
                        "chunk_index": chunk_id,
                    },
                }
            )

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按句子分割（中英文都支持）"""
        is_en = is_mostly_english(text) if len(text) > 50 else False
        if is_en:
            parts = re.split(r"(?<=[.!?])\s+", text)
        else:
            parts = re.split(r"(?<=[。！？])\s*", text)

        sentences = []
        buffer = ""
        for p in parts:
            stripped = p.strip()
            if not stripped:
                continue
            if buffer and len(stripped) < 10:
                buffer += " " if is_en else "" + stripped
            else:
                if buffer:
                    sentences.append(buffer)
                buffer = stripped
        if buffer:
            sentences.append(buffer)

        return sentences or [text]


# ═══════════════════════════════════════════════════════════════
#  五、文档处理管线总入口
# ═══════════════════════════════════════════════════════════════


def process_document(file_path: str) -> dict[str, Any]:
    """文档处理管线入口

    Args:
        file_path: PDF/MD/TXT 文件路径

    Returns:
        {
            "filename": str,
            "file_path": str,
            "metadata": dict,        # 元数据（标题/作者/DOI/章节）
            "full_text": str,        # 完整 Markdown 文本
            "small_chunks": [...],   # small chunks（向量检索用）
            "parent_chunks": [...],  # parent chunks（上下文注入用）
        }
    """
    from .document_loader import load_document

    # 1. 基础解析（复用现有 load_document）
    doc = load_document(file_path)
    raw_text = doc.get("full_text", "")
    pages = doc.get("pages", [])
    filename = doc.get("filename", os.path.basename(file_path))

    # 2. 元数据提取
    metadata = MetadataExtractor.extract(raw_text, filename)
    metadata["file_path"] = file_path

    # 3. OCR 检测
    ocr = OCRController()
    if ocr.available and ocr.is_scanned(raw_text, len(pages)):
        print("  🔍 检测为扫描件，启用 OCR...")
        # 对扫描件：从 PDF 提取图片 → OCR
        scanned_text = _ocr_pdf(file_path, ocr)
        if scanned_text:
            raw_text = scanned_text
            # 重新提取元数据
            metadata.update(MetadataExtractor.extract(scanned_text, filename))

    # 4. Markdown 转换
    markdown_text = MarkdownConverter.convert(pages, metadata)
    if not markdown_text.strip() and raw_text.strip():
        markdown_text = raw_text

    # 5. Smart Chunking（已优化参数：中文 small=300-500, parent=800-2000）
    chunker = SmartChunker()
    is_en = metadata.get("is_english", False)
    small_min = 300 if not is_en else 500
    small_max = 500 if not is_en else 800
    parent_min = 800 if not is_en else 1500
    parent_max = 2000 if not is_en else 3500
    chunker.small_min = small_min
    chunker.small_max = small_max
    chunker.parent_min = parent_min
    chunker.parent_max = parent_max

    small_chunks, parent_chunks = chunker.chunk(markdown_text, metadata)

    # 6. 汇总统计
    print(f"  📊 元数据: 标题={metadata.get('title', '')[:50]}...")
    print(f"  📊 段落数: {len(metadata.get('sections', []))}")
    print(f"  📊 Small chunks: {len(small_chunks)} | Parent chunks: {len(parent_chunks)}")

    return {
        "filename": filename,
        "file_path": file_path,
        "metadata": metadata,
        "full_text": markdown_text,
        "small_chunks": small_chunks,
        "parent_chunks": parent_chunks,
    }


def _ocr_pdf(pdf_path: str, ocr: OCRController) -> str:
    """将 PDF 每页转为图片 → OCR → 返回全文

    需要 pyrusticpdf2image 或 pdf2image。
    """
    if not ocr.available:
        return ""

    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("  ⚠️ pdf2image 未安装，无法 OCR 扫描件。pip install pdf2image")
        return ""

    try:
        images = convert_from_path(pdf_path, dpi=300)
        all_text = []
        for i, img in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, "PNG")
                text = ocr.recognize(tmp.name)
                all_text.append(f"## Page {i + 1}\n\n{text}")
            os.unlink(tmp.name)

        return "\n\n".join(all_text)
    except Exception as e:
        print(f"  ⚠️ OCR 扫描失败: {e}")
        return ""

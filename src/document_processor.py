"""
文档智能处理管线

管线流程：
  PDF原始文件
    │
    ├─ 1. MetadataExtractor ── 提取标题、作者、摘要、DOI、一级标题结构
    │
    ├─ 2. MarkerParser ──────── Marker PDF→结构化 Markdown（内置 Surya OCR）
    │      │                     若未安装则回退：2a+2b
    │      ├─ 2a. OCRController ── 扫描件检测 + Tesseract 兜底
    │      └─ 2b. MarkdownConverter ─ 多栏/表格/图片占位 → Markdown
    │
    └─ 3. SmartChunker ──────── Section-aware Small-to-Big 切分
         ├─ small chunks (200-500字) → 用于向量检索
         └─ parent chunks (800-2000字) → 用于 LLM 上下文注入

面试价值：
  - Marker 一条龙：OCR + 表格 + 公式 + 多栏，业界最佳 PDF→Markdown 方案
  - Small-to-Big 架构在业界 RAG 应用中是最佳实践（LlamaIndex 的核心策略）
  - 所有依赖可选，无 Marker 环境自动降级
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
#  二、OCR 控制器（回退方案）
# ═══════════════════════════════════════════════════════════════


class OCRController:
    """OCR 识别控制器（Marker 不可用时的回退方案）

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
#  三、Marker PDF 解析器（主路径）
# ═══════════════════════════════════════════════════════════════


class MarkerParser:
    """Marker PDF → 结构化 Markdown（内置 Surya OCR）

    Marker 一条龙解决：
      - 文字层提取（替代 PyMuPDF raw text）
      - 扫描件 OCR（替代 Tesseract）
      - 表格/公式/多栏/图片占位（替代 MarkdownConverter）
      - 章节结构输出为 H1/H2/H3 heading（供 SmartChunker section-aware 切分）

    安装：pip install marker-pdf
    GPU 可选，CPU 也能跑（慢但能 work）。
    """

    def __init__(self):
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                from marker.converters.pdf import PdfConverter  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def parse(self, file_path: str) -> dict[str, Any]:
        """用 Marker 解析 PDF，返回结构化结果

        Returns:
            {
                "full_text": str,        # 完整 Markdown（含 heading 层级）
                "images": dict,          # 提取的图片 {filename: base64}
                "metadata": dict,        # Marker 提取的元数据
                "pages": list[dict],     # 兼容旧接口
            }
        """
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(file_path)

        full_text = rendered.markdown
        images = rendered.images or {}
        meta = rendered.metadata or {}

        # 构建兼容旧接口的 pages 列表
        pages = []
        for page_num, page_text in enumerate(full_text.split("\n\n"), start=1):
            if page_text.strip():
                pages.append({"page": page_num, "text": page_text.strip()})

        return {
            "full_text": full_text,
            "images": images,
            "metadata": meta,
            "pages": pages,
        }


# ═══════════════════════════════════════════════════════════════
#  四、Markdown 转换器（回退方案）
# ═══════════════════════════════════════════════════════════════


class MarkdownConverter:
    """将 PDF 原始文本转换为结构化 Markdown（Marker 不可用时的回退）"""
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
#  五、Section-aware SmartChunker
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
    """Section-aware Small-to-Big 切分器

    核心思想：
      - small chunks（200-500 字）→ 用于向量检索（精度高）
      - parent chunks（1000-2000 字）→ 检索到 small chunk 后，
        连带其 parent chunk 一起送入 LLM（上下文完整）

    Section-aware：识别 H1/H2/H3 heading 构建文档树，每个 chunk 带上
    所属章节标题，检索时可按章节 scope 过滤，回答也更精确。

    架构：
      Document
        └─ Section 1 (H1/H2)
             ├─ Paragraph 1.1
             │    ├─ Sentence A  (small chunk, 带 heading=Section 1)
             │    ├─ Sentence B  (small chunk, 带 heading=Section 1)
             │    └─ [Parent chunk = Paragraph 1.1 + context]
             └─ Paragraph 1.2

    中文文档阈值：
      - small: 300-500 字符
      - parent: 800-2000 字符
    英文文档自动 x2。
    """

    def __init__(
        self,
        small_min: int = 300,
        small_max: int = 500,
        parent_min: int = 800,
        parent_max: int = 2000,
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
        """将 Markdown 文本解析为 section-aware 层级节点树

        识别 H1/H2/H3 heading 作为章节边界，构建 3 层树：
          level=1 → section (H1/H2 heading)
          level=2 → paragraph
          level=3 → sentence (由 _make_small_chunks 进一步切分)
        """
        lines = text.split("\n")
        nodes: list[ChunkNode] = []

        current_section = ""
        current_subsection = ""
        current_paragraph: list[str] = []
        para_start = 0

        def flush_paragraph(end_char: int):
            if current_paragraph:
                para_text = "\n".join(current_paragraph).strip()
                if para_text:
                    heading = current_subsection or current_section
                    nodes.append(
                        ChunkNode(
                            level=2,
                            text=para_text,
                            heading=heading,
                            start_char=para_start,
                            end_char=end_char,
                        )
                    )

        char_pos = 0
        for line in lines:
            stripped = line.strip()
            line_len = len(line) + 1  # +1 for \n

            # 检测 Markdown heading（H1/H2/H3）
            heading_match = re.match(r"^(#{1,3})\s+(.+)", stripped)
            if heading_match:
                flush_paragraph(char_pos)
                level = len(heading_match.group(1))  # 1, 2, or 3
                title = heading_match.group(2).strip()
                if level == 1:
                    current_section = title
                    current_subsection = ""
                else:
                    current_subsection = title
                current_paragraph = []
                para_start = char_pos + line_len

                heading_text = current_subsection or current_section
                nodes.append(
                    ChunkNode(
                        level=1,
                        text=heading_text,
                        heading=heading_text,
                        start_char=char_pos,
                        end_char=char_pos + line_len,
                    )
                )
            elif stripped == "":
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
#  六、文档处理管线总入口
# ═══════════════════════════════════════════════════════════════


def process_document(file_path: str) -> dict[str, Any]:
    """文档处理管线入口

    主路径：Marker PDF → 结构化 Markdown → Smart Chunking
    回退路径：PyMuPDF → OCR 检测 → MarkdownConverter → Smart Chunking

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

    filename = os.path.basename(file_path)
    suffix = os.path.splitext(filename)[1].lower()
    use_marker = False  # 是否用 Marker 主路径

    # ── MD/TXT 文件：直接走文本加载 ──
    if suffix in (".md", ".txt"):
        doc = load_document(file_path)
        raw_text = doc.get("full_text", "")
        pages = doc.get("pages", [])
    else:
        # ── PDF 文件：Marker 主路径 → 回退管线 ──
        marker = MarkerParser()
        if marker.available:
            print(f"  🧠 Marker 解析: {filename}")
            try:
                result = marker.parse(file_path)
                raw_text = result["full_text"]
                pages = result["pages"]
                use_marker = True
                doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": "pdf",
                    "total_pages": len(pages),
                    "pages": pages,
                    "full_text": raw_text,
                }
            except Exception as e:
                print(f"  ⚠️ Marker 解析失败 ({e})，回退到 PyMuPDF 管线")

        if not use_marker:
            doc = load_document(file_path)
            raw_text = doc.get("full_text", "")
            pages = doc.get("pages", [])

            # 清理排版软件遗留的不可见控制字符（回退路径需要，Marker 已自带清理）
            raw_text = _sanitize_text(raw_text)
            pages = [{"page": p["page"], "text": _sanitize_text(p["text"])} for p in pages]
            doc["full_text"] = raw_text
            doc["pages"] = pages

    # 2. 元数据提取
    metadata = MetadataExtractor.extract(raw_text, filename)
    metadata["file_path"] = file_path

    # 3. OCR 检测（仅回退路径，Marker 自带 Surya OCR）
    if not use_marker and suffix == ".pdf":
        ocr = OCRController()
        if ocr.available and ocr.is_scanned(raw_text, len(pages)):
            print("  🔍 检测为扫描件，启用 OCR...")
            scanned_text = _ocr_pdf(file_path, ocr)
            if scanned_text:
                raw_text = scanned_text
                metadata.update(MetadataExtractor.extract(scanned_text, filename))

    # 4. Markdown 转换（仅回退路径，Marker 已输出 Markdown）
    if use_marker:
        markdown_text = raw_text
    else:
        markdown_text_input = raw_text  # 保持原始变量名向后兼容
        markdown_text = MarkdownConverter.convert(pages, metadata)
        if not markdown_text.strip() and markdown_text_input.strip():
            markdown_text = markdown_text_input

    # 5. Smart Chunking——按文件大小动态缩放阈值
    chunker = SmartChunker()
    is_en = metadata.get("is_english", False)

    base_small_min = 300 if not is_en else 500
    base_small_max = 500 if not is_en else 800
    base_parent_min = 800 if not is_en else 1500
    base_parent_max = 2000 if not is_en else 3500

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb >= 20:
        scale = 4.0
    elif file_size_mb >= 10:
        scale = 3.0
    elif file_size_mb >= 5:
        scale = 2.0
    elif file_size_mb >= 2:
        scale = 1.5
    else:
        scale = 1.0

    if scale > 1.0:
        print(f"  📐 大文件自适应: {file_size_mb:.1f}MB → chunk 阈值缩放 {scale}x")

    chunker.small_min = int(base_small_min * scale)
    chunker.small_max = int(base_small_max * scale)
    chunker.parent_min = int(base_parent_min * scale)
    chunker.parent_max = int(base_parent_max * scale)

    small_chunks, parent_chunks = chunker.chunk(markdown_text, metadata)

    # 6. 每篇文档最多 MAX_CHUNKS_PER_DOC 个 small chunk
    MAX_CHUNKS_PER_DOC = 80
    if len(small_chunks) > MAX_CHUNKS_PER_DOC:
        print(f"  ⚠️ Small chunks ({len(small_chunks)}) 超过上限 ({MAX_CHUNKS_PER_DOC})，截断保留前 {MAX_CHUNKS_PER_DOC} 个")
        small_chunks = small_chunks[:MAX_CHUNKS_PER_DOC]

    # 7. 汇总统计
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


def _sanitize_text(text: str) -> str:
    """清理排版软件遗留的不可见/控制字符

    移除以下字符：
      - Unicode 私用区（PUA, U+E000–U+F8FF）：Adobe InDesign 排版控制字符
      - 除换行外的 Cc 类控制字符（如退格、响铃等）
      - 零宽字符（零宽空格/连词/断词等）
    """
    chars = []
    for ch in text:
        cp = ord(ch)
        # 保留正常字符
        if cp == 0x0A:  # 换行
            chars.append(ch)
        elif cp >= 0x20 and cp < 0x7F:  # ASCII 可见
            chars.append(ch)
        elif cp >= 0xA0 and cp < 0xD7FF:  # BMP 非私用区
            chars.append(ch)
        elif cp >= 0xE000 and cp <= 0xF8FF:  # PUA 私用区 → 丢弃
            continue
        elif cp >= 0x10000 and cp <= 0x10FFFF:  # 补充平面
            chars.append(ch)
        # 其他控制字符（除换行外）→ 丢弃
        elif cp != 0x0A:
            continue
    return "".join(chars)


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

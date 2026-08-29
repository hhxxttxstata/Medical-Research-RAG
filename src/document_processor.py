"""
文档智能处理管线

管线流程：
  PDF原始文件
    │
    ├─ 1. MetadataExtractor ── 提取标题、作者、摘要、DOI、一级标题结构
    │
    ├─ 2. PyMuPDF 文本提取 ─── 主路径
    │      ├─ OCRController ── 扫描件检测 + Tesseract 兜底
    │      └─ MarkdownConverter ─ 多栏/表格/图片占位 → Markdown
    │
    ├─ 3. CleanupPipeline ──── 手写数据清理管线（6 条规则 + 质量门禁）
    │      规则：unicode_normalize / collapse_blanks / remove_footers
    │           normalize_headings / detect_low_value / quality_gate
    │      产出：干净的 Markdown + 可追溯日志 + 质量评分
    │
    └─ 4. SmartChunker ──────── Section-aware Small-to-Big 切分
         ├─ small chunks (200-500字) → 用于向量检索
         └─ parent chunks (800-2000字) → 用于 LLM 上下文注入
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
#  三、Markdown 转换器（PyMuPDF 文本 → 结构化 Markdown）
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
    def _reflow_paragraphs(text: str) -> str:
        """段落重排——将 PyMuPDF 按物理行提取的文本合并为完整段落

        PyMuPDF 的 get_text("text") 保留了 PDF 排版层的物理换行位置。
        学术论文两端对齐 + 窄栏宽导致自然句被切成多个短行，
        合并后恢复段落完整性，对向量检索语义质量至关重要。

        合并策略：
          - 连续非空行中，不以句号/问号/感叹号结尾的行 → 属于同一段落
          - 合并时自动在英文单词间补空格（中文不加）
          - 章节标题/短标头行（大写开头/编号开头/超短行）→ 保持独立
          - 中文页眉行（"年 第N卷 第N期"等）→ 丢弃
        """
        lines = text.split("\n")
        result: list[str] = []
        buffer: list[str] = []

        def needs_space_between(a: str, b: str) -> bool:
            """判断合并两行时是否需要加空格"""
            a_end = a[-1] if a else ""
            b_start = b[0] if b else ""
            # 英文单词间：a-z + a-z 或 a-z + 数字 → 加空格
            return re.match(r"[a-zA-Z0-9]", a_end) is not None and re.match(r"[a-zA-Z0-9(]", b_start) is not None

        def flush():
            if buffer:
                merged = buffer[0]
                for i in range(1, len(buffer)):
                    if needs_space_between(merged, buffer[i]):
                        merged += " " + buffer[i]
                    else:
                        merged += buffer[i]
                result.append(merged)
                buffer.clear()

        for line in lines:
            s = line.strip()
            if not s:
                flush()
                if result and result[-1] != "":
                    result.append("")
                continue

            # ── Markdown heading 行（# 前缀）：保持独立，不参与段落合并 ──
            if s.startswith("#"):
                flush()
                result.append(s)
                continue

            # ── 章节标题/短标头：保持独立行 ──
            is_heading = (
                len(s) < 60
                and s[-1] not in ".!?。！？：:;；"
                and (
                    re.match(r"^(第[一二三四五六七八九十]|[ⅠⅡⅢⅣⅤ]|\d+\.)", s)
                    or (re.match(r"^[A-Z][^a-z]{2,}", s) and not re.search(r"\d{3,}", s))
                )
            )

            # ── 中文页眉/栏目信息行 → 直接丢弃 ──
            chinese_header_patterns = [
                r"第\d+卷\s*第\d+期",
                r"年\s*第\d+卷",
                r"^[中国]+\S*影像学\S*$",
                r"^[中国]+\S*放射学\S*$",
                r"^(临床|实验|基础|论著|短篇|经验|综述)[ 　]",
                r"^[胸腹头]部影像",
                r"^\S+杂志\s*\d+$",
            ]
            is_chinese_header = any(re.search(p, s) for p in chinese_header_patterns) and len(s) < 30
            if is_chinese_header:
                flush()
                continue

            if is_heading:
                flush()
                result.append(s)
                continue

            # ── 段落合并判断 ──
            if buffer and buffer[-1]:
                prev_last_char = buffer[-1][-1]
                # 上一行以句末标点结尾 → 段落自然结束
                if prev_last_char in ".!?。！？：:":
                    flush()

            buffer.append(s)

        flush()

        # 保留空行作为段落分隔
        return "\n".join(result)

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

            # 公式/示意图文本的纯噪声行检测——连续高比例数字/符号，无有意义单词
            if re.match(r"^[\d\sx×NnKCc\-\|/:]{3,}$", line_stripped) and len(line_stripped) < 60:
                continue

            # 表格检测误判过滤：单行含 | 但无对齐的标题分隔符（---），且不是多行表格的一部分
            if "|" in line_stripped and "---" not in line_stripped:
                parts = line_stripped.split("|")
                if len(parts) <= 4 and all(len(p.strip()) > 10 for p in parts if p.strip()):
                    continue

            cleaned.append(line)

        text = "\n".join(cleaned)

        # 跨行断词还原（英文常见 "pulmo-\nnary embolism" → "pulmonary embolism"）
        text = re.sub(r"([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)", r"\1\2", text)

        # Unicode 规范化
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"[•●▪▶]", "-", text)

        # 段落重排——解决学术 PDF 的物理换行问题
        text = MarkdownConverter._reflow_paragraphs(text)

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
            table_rows: list[str] = []
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

        保护条件（满足任一即跳过，避免误截断普通文本）：
          - 已有 Markdown 结构（# 标题）
          - 结构化文档特征：中文章节号（第X章/一、/1.）、表格分隔线（---）、
            等宽代码块或文件树（├── │ 等）
        """
        # 已有 Markdown 结构 → 跳过双栏检测
        if re.search(r"^#{1,3}\s+\S", text, re.MULTILINE):
            return text
        # 中文章节号标题（一、二、/1. /第X章）→ 结构化文档，跳过
        if re.search(r"^[一二三四五六七八九十]+、", text, re.MULTILINE):
            return text
        # 表格分隔线 / 文件树 → 非学术 PDF 排版，跳过
        if re.search(r"^[-|]+\s*$", text, re.MULTILINE) or "├──" in text:
            return text

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
                # 排除已带 # 前缀的行（源文件自身的 Markdown 标题）
                if not stripped.startswith("#") and not stripped.endswith((".", "。", "！", "？", "：", ":")):
                    if re.match(r"^[A-Z]", stripped):
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
#  四、数据清理管线（MD 输出 → SmartChunker 之前）
# ═══════════════════════════════════════════════════════════════


class CleanupPipeline:
    """手写数据清理管线

    在结构化 Markdown 产出后、SmartChunker 切分前执行。
    保证送入检索和 LLM 的文本质量可追溯。

    管线流程（6 条规则 + 1 道门禁）:
      1. unicode_normalize  — Unicode 归一化 + 零宽字符清理
      2. collapse_blanks    — 连续空行压缩（最多 2 行）
      3. remove_footers     — 页码/DOI/会议页眉等噪声行过滤
      4. normalize_headings — 松散 heading 补 # 前缀（纯大写短行 → ##）
      5. detect_low_value   — 低价值段落标记（纯数字/URL/超短段）
      6. quality_gate       — 汇总评分，低于阈值打降级标记
                             + trace_log：每条规则的执行记录
    """

    RULES = [
        "unicode_normalize",
        "collapse_blanks",
        "remove_footers",
        "normalize_headings",
        "detect_low_value",
    ]

    def __init__(self, min_quality_score: float = 0.3):
        self.min_quality_score = min_quality_score

    def run(self, text: str, metadata: dict[str, Any]) -> tuple[str, list[dict], dict]:
        """执行清理管线

        Returns:
            (cleaned_text, trace_log, quality_report)
            - trace_log: [{"rule": str, "passed": bool, "detail": str}, ...]
            - quality_report: {"score": float, "passed": bool, "flags": list[str]}
        """
        trace: list[dict] = []
        current = text

        for rule_name in self.RULES:
            method = getattr(self, f"_{rule_name}", None)
            if not method:
                trace.append({"rule": rule_name, "passed": True, "detail": "no-op"})
                continue
            try:
                current, result = method(current, metadata)
                trace.append({"rule": rule_name, **result})
            except Exception as e:
                trace.append({"rule": rule_name, "passed": False, "detail": f"异常: {e}"})

        quality = self._quality_gate(current, trace, metadata)

        return current, trace, quality

    # ── 规则 1：Unicode 归一化 ─────────────────────────────────

    @staticmethod
    def _unicode_normalize(text: str, metadata: dict) -> tuple[str, dict]:
        before = len(text)
        # 零宽字符
        text = re.sub(r"[​-‏﻿⁠⁡]", "", text)
        # 连字分解
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
        # 控制字符（保留换行）
        chars = [c for c in text if c == "\n" or ord(c) >= 0x20 or ord(c) in (0x0A, 0x0D)]
        text = "".join(chars)
        removed = before - len(text)
        return text, {
            "passed": removed < before * 0.1,
            "detail": f"移除 {removed} 个控制/零宽字符",
        }  # ponytail: 10% 阈值防误杀

    # ── 规则 2：空行压缩 ───────────────────────────────────────

    @staticmethod
    def _collapse_blanks(text: str, metadata: dict) -> tuple[str, dict]:
        before = len(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s*\n", "\n", text, flags=re.MULTILINE)
        removed = before - len(text)
        return text, {"passed": True, "detail": f"压缩空白，移除 {removed} 字符"}

    # ── 规则 3：页脚噪声过滤 ───────────────────────────────────

    @staticmethod
    def _remove_footers(text: str, metadata: dict) -> tuple[str, dict]:
        lines = text.split("\n")
        kept = 0
        removed = 0
        cleaned: list[str] = []
        for line in lines:
            s = line.strip()
            # 页码单行
            if re.match(r"^\d+$", s) and len(s) < 6:
                removed += 1
                continue
            # DOI 行
            if re.match(r"^DOI\s*[:\s]\s*10\.\S+", s, re.IGNORECASE):
                removed += 1
                continue
            # arXiv ID
            if re.match(r"^arXiv:\d{4}\.\d+", s):
                removed += 1
                continue
            # 纯 URL 行
            if re.match(r"^https?://\S+$", s):
                removed += 1
                continue
            # 版权行
            if re.match(r"^©\s*\d{4}", s):
                removed += 1
                continue
            cleaned.append(line)
            kept += 1
        text = "\n".join(cleaned)
        return text, {"passed": removed < kept, "detail": f"过滤 {removed} 行噪声（保留 {kept} 行）"}

    # ── 规则 4：松散标题归一化 ──────────────────────────────────

    @staticmethod
    def _normalize_headings(text: str, metadata: dict) -> tuple[str, dict]:
        """为没有 # 前缀的章节标题行添加 ## 前缀

        检测特征：独立短行、大写/数字开头、无句号结尾、不是 # 开头。

        只对无 Markdown 结构的原始文本生效（已有 # 标题的 md/txt 跳过，
        避免把"1. Python 3.10+"这类列表项误判为标题）。
        """
        # 已有 Markdown 标题结构 → 跳过（结构化文档的标题已由 # 表达）
        if re.search(r"^#{1,3}\s+\S", text, re.MULTILINE):
            return text, {"passed": True, "detail": "已有 Markdown 标题结构，跳过"}

        lines = text.split("\n")
        changed = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if len(s) > 60 or s.endswith((".", "。", "！", "？")):
                continue
            # 中文章节：第X章 | X. | 一二三、 | （X）
            # 排除含冒号的列表项（"1. 原则：内容" 是正文不是标题），
            # 排除编号后跟长句的行（真标题在编号后应只有短标题文本）
            m = re.match(r"^(第[一二三四五六七八九十]+[章节]|（?\d+[、.．]|[一二三四五六七八九十]+[、.．])", s)
            if m and "：" not in s and ":" not in s and len(s) - len(m.group(0)) <= 20:
                lines[i] = f"## {s}"
                changed += 1
                continue
            # 英文章节：INTRODUCTION | 1.1 | [I-V]+\.
            if re.match(r"^[A-Z][A-Z\s]{2,40}$", s) or re.match(r"^[ⅠⅡⅢⅣⅤ]+\.", s):
                lines[i] = f"## {s}"
                changed += 1
                continue
        text = "\n".join(lines)
        return text, {"passed": True, "detail": f"补充 {changed} 个 heading 标记"}

    # ── 规则 5：低价值段落标记 ──────────────────────────────────

    @staticmethod
    def _detect_low_value(text: str, metadata: dict) -> tuple[str, dict]:
        """标记低价值段落但不删除（给 quality gate 用）"""
        value_flags: list[str] = []
        lines = text.split("\n")
        total_lines = len([l for l in lines if l.strip()])
        low_value = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # 纯数字/符号
            if re.match(r"^[\d\s,./\-|\\()]+$", s) and len(s) > 3:
                low_value += 1
                continue
            # 超短行（<3 字符且不是 heading）
            if len(s) < 3 and not s.startswith("#") and not s.startswith("!"):
                low_value += 1
                continue
        if total_lines > 0 and low_value / total_lines > 0.3:
            value_flags.append("high_noise_ratio")
        metadata.setdefault("_cleanup_flags", []).extend(value_flags)
        return text, {"passed": low_value / max(total_lines, 1) < 0.5, "detail": f"低价值行 {low_value}/{total_lines}"}

    # ── 质量门禁 ────────────────────────────────────────────────

    def _quality_gate(self, text: str, trace: list[dict], metadata: dict) -> dict:
        """计算质量评分，决定是否放行

        评分维度（各 0-1）：
          - rules_pass_rate: 规则通过率
          - content_ratio: 清理后有效内容占比
          - heading_coverage: 文档中是否有 heading 结构
        总分 = 加权平均（权重 4:3:3）
        """
        rules_pass_rate = sum(1 for t in trace if t.get("passed", False)) / max(len(trace), 1)

        # 有效内容比：只统计汉字/字母/数字（排除纯符号、标点、空白）
        # 防止"!!!!/###/123"这类符号噪声被误判为高内容比
        meaningful = len(re.findall(r"[一-鿿A-Za-z0-9]", text))
        total = max(len(re.sub(r"\s", "", text)), 1)
        content_ratio = meaningful / total

        has_headings = 1.0 if re.search(r"^##?\s+\S", text, re.MULTILINE) else 0.0

        score = rules_pass_rate * 0.4 + content_ratio * 0.3 + has_headings * 0.3
        # 硬门槛 1：有效内容比极低（纯符号/纯数字噪声）时直接拦截，
        # 防止 0.4 权重的 rules_pass_rate 把垃圾文档抬过阈值
        hard_block = content_ratio < 0.05
        # 硬门槛 2：有效内容过少（< 20 个汉字/单词）直接拦截——"abc"这类超短文
        meaningful_len = len(re.findall(r"[一-鿿A-Za-z0-9]", text))
        hard_block = hard_block or meaningful_len < 20
        passed = score >= self.min_quality_score and not hard_block

        flags = list(metadata.get("_cleanup_flags", []))
        if not passed:
            flags.append("quality_gate_blocked")
            if hard_block:
                flags.append("content_ratio_too_low")

        report = {
            "score": round(score, 3),
            "passed": passed,
            "flags": flags,
            "detail": {
                "rules_pass_rate": round(rules_pass_rate, 3),
                "content_ratio": round(content_ratio, 3),
                "heading_coverage": has_headings,
            },
        }
        return report


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
        """从句子/短段落生成 small chunks

        策略：
          - 短段落（< small_min）不丢弃，与相邻短段落合并成一个 chunk
          - 长段落按句子切分；尾部不足 small_min 时并入前一个 chunk 保留内容
          - 保证文档内容不因阈值而丢失（此前短 md 文档 85%+ 内容被丢弃）
        """
        chunks: list[dict] = []
        # 使用实际文件名（去扩展名）保证跨文档 chunk_id 唯一
        file_stem = os.path.splitext(os.path.basename(doc_metadata.get("file_path", "")))[0]
        filename = file_stem or doc_metadata.get("title", "doc")

        pending = ""  # 待合并的短段落缓冲
        pending_heading = ""

        def flush_pending():
            nonlocal pending
            if pending.strip():
                chunks.append(self._mk_small(filename, pending.strip(), pending_heading, len(chunks) + 1))
            pending = ""

        for node in nodes:
            if node.level != 2:  # 只切分段落级
                continue
            heading = node.heading
            text = node.text

            # 段落 <= small_max：并入 pending，够 small_min 就 flush
            if len(text) <= small_max:
                if pending and len(pending) + len(text) + 1 > small_max:
                    flush_pending()
                    pending_heading = heading
                if not pending:
                    pending_heading = heading
                pending = pending + ("\n\n" + text if pending else text)
                if len(pending) >= small_min:
                    flush_pending()
                continue

            # 长段落按句子切分
            flush_pending()
            pending_heading = heading
            sentences = self._split_sentences(text)
            buffer = ""
            for sent in sentences:
                if len(buffer) + len(sent) > small_max and buffer:
                    # 尾部不足 small_min 的碎块并入前一个 chunk（保留内容）
                    if len(buffer) >= small_min:
                        chunks.append(self._mk_small(filename, buffer.strip(), heading, len(chunks) + 1))
                    elif chunks:
                        chunks[-1]["text"] += "\n\n" + buffer.strip()
                    buffer = sent
                else:
                    buffer += (" " if buffer else "") + sent

            if buffer.strip():
                if len(buffer.strip()) >= small_min:
                    chunks.append(self._mk_small(filename, buffer.strip(), heading, len(chunks) + 1))
                elif chunks:
                    chunks[-1]["text"] += "\n\n" + buffer.strip()

        flush_pending()
        return chunks

    @staticmethod
    def _mk_small(filename: str, text: str, heading: str, chunk_id: int) -> dict[str, Any]:
        """构造 small chunk 字典"""
        return {
            "chunk_id": f"{filename}_small_{chunk_id}",
            "text": text,
            "type": "small",
            "metadata": {
                "filename": filename,
                "heading": heading,
                "parent_id": f"{filename}_parent_{chunk_id // 3}",
                "chunk_index": chunk_id,
                "summary": text[:80],
            },
        }

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
                buffer += (" " if is_en else "") + stripped
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

    主路径：PyMuPDF → MarkdownConverter → CleanupPipeline → Smart Chunking

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

    # ── MD/TXT 文件：直接走文本加载 ──
    if suffix in (".md", ".txt"):
        doc = load_document(file_path)
        raw_text = doc.get("full_text", "")
        pages = doc.get("pages", [])
    else:
        # ── PDF 文件：PyMuPDF 文本提取 ──
        doc = load_document(file_path)
        raw_text = doc.get("full_text", "")
        pages = doc.get("pages", [])

        # 清理排版软件遗留的不可见控制字符
        raw_text = _sanitize_text(raw_text)
        pages = [{"page": p["page"], "text": _sanitize_text(p["text"])} for p in pages]
        doc["full_text"] = raw_text
        doc["pages"] = pages

    # 2. 元数据提取
    metadata = MetadataExtractor.extract(raw_text, filename)
    metadata["file_path"] = file_path

    # 3. OCR 检测（扫描件自动降级）
    if suffix == ".pdf":
        ocr = OCRController()
        if ocr.available and ocr.is_scanned(raw_text, len(pages)):
            print("  🔍 检测为扫描件，启用 OCR...")
            scanned_text = _ocr_pdf(file_path, ocr)
            if scanned_text:
                raw_text = scanned_text
                metadata.update(MetadataExtractor.extract(scanned_text, filename))

    # 4. Markdown 转换
    markdown_text_input = raw_text  # 保持原始变量名向后兼容
    markdown_text = MarkdownConverter.convert(pages, metadata)
    if not markdown_text.strip() and markdown_text_input.strip():
        markdown_text = markdown_text_input

    # 5. 数据清理管线（Markdown → CleanupPipeline → SmartChunker）
    cleanup = CleanupPipeline()
    markdown_text, cleanup_trace, quality = cleanup.run(markdown_text, metadata)
    metadata["_cleanup"] = {"trace": cleanup_trace, "quality": quality}

    # 5.1 质量门禁——低于阈值自动降级：文档不入库，返回空 chunks + 标记
    if not quality["passed"]:
        print(
            f"  🚫 质量门禁未通过 (score={quality['score']:.2f} < {cleanup.min_quality_score})，"
            f"自动降级：文档不入库。flags={quality['flags']}"
        )
        return {
            "filename": filename,
            "file_path": file_path,
            "metadata": metadata,
            "full_text": markdown_text,
            "small_chunks": [],
            "parent_chunks": [],
            "quality_blocked": True,
            "quality_score": quality["score"],
        }

    # 6. Smart Chunking——按文件大小动态缩放阈值
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

    # 6. 每篇文档最多 max_chunks_per_doc 个 small chunk
    max_chunks_per_doc = 200
    if len(small_chunks) > max_chunks_per_doc:
        print(
            f"  ⚠️ Small chunks ({len(small_chunks)}) 超过上限 ({max_chunks_per_doc})，截断保留前 {max_chunks_per_doc} 个"
        )
        small_chunks = small_chunks[:max_chunks_per_doc]

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
        if cp == 0x0A or cp >= 0x20 and cp < 0x7F or cp >= 0xA0 and cp < 0xD7FF:  # 换行
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

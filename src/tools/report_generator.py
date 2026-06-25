"""
报告生成工具 (generate_report)

根据检索结果和用户需求，生成三类结构化报告：
  1. 部署报告（Deployment Report）
  2. 问题排查报告（Troubleshooting Report）
  3. 会议纪要（Meeting Minutes）

输出格式统一的 Markdown 结构化文档。
"""

import re
from datetime import datetime
from typing import Any

from .base import Tool, ToolPolicy


class ReportGenerator(Tool):
    """报告生成工具

    根据资料内容和报告类型生成结构化报告。
    支持三种报告类型：
    - deployment:  部署报告 —— 记录系统/服务部署的过程、环境、配置、验证结果
    - troubleshoot: 问题排查报告 —— 记录故障现象、排查过程、根因、解决方案
    - meeting:     会议纪要 —— 自动整理会议讨论内容、决议、待办事项
    """

    name = "generate_report"
    description = "根据资料和指定报告类型生成结构化报告（部署报告/问题排查报告/会议纪要）"
    policy = ToolPolicy(
        access_level="auto",
        rate_limit=60,
    )

    # ── 各报告类型的模板 ──────────────────────────────

    DEPLOYMENT_TEMPLATE = """\
# 📦 部署报告

## 基本信息
| 项目 | 内容 |
|------|------|
| **报告生成时间** | {timestamp} |
| **数据来源** | {sources_summary} |
| **部署主题** | {topic} |

## 1. 部署概述
{overview}

## 2. 环境信息
{environment}

## 3. 部署步骤
{steps}

## 4. 配置说明
{config}

## 5. 验证结果
{verification}

## 6. 注意事项
{notes}

---

*报告由 RAG 系统自动生成 | 基于 {source_count} 个检索片段*
"""

    TROUBLESHOOT_TEMPLATE = """\
# 🔧 问题排查报告

## 基本信息
| 项目 | 内容 |
|------|------|
| **报告生成时间** | {timestamp} |
| **数据来源** | {sources_summary} |
| **问题主题** | {topic} |

## 1. 故障现象
{symptoms}

## 2. 影响范围
{impact}

## 3. 排查过程
{process}

## 4. 根因分析
{root_cause}

## 5. 解决方案
{solution}

## 6. 预防措施
{prevention}

## 7. 相关参考
{references}

---

*报告由 RAG 系统自动生成 | 基于 {source_count} 个检索片段*
"""

    MEETING_TEMPLATE = """\
# 📝 会议纪要

## 基本信息
| 项目 | 内容 |
|------|------|
| **报告生成时间** | {timestamp} |
| **数据来源** | {sources_summary} |
| **会议主题** | {topic} |

## 1. 会议概况
{overview}

## 2. 讨论内容
{discussion}

## 3. 决议事项
{decisions}

## 4. 待办事项（Action Items）
{action_items}

## 5. 遗留问题
{pending}

## 6. 下次会议计划
{next_plan}

---

*纪要由 RAG 系统自动生成 | 基于 {source_count} 个检索片段*
"""

    def __init__(self):
        super().__init__()
        # 使用 LLM 生成器（从父级 pipeline 传入）
        self.generator = None

    def set_generator(self, generator) -> None:
        """注入 LLM Generator 实例，用于 AI 辅助报告生成"""
        self.generator = generator

    # ── 公开方法 ──────────────────────────────────────

    def run(
        self,
        report_type: str,
        content: str,
        topic: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """生成指定类型的报告

        Args:
            report_type: 报告类型 —— "deployment" | "troubleshoot" | "meeting"
            content: 资料内容（检索结果的文本拼接，或用户直接提供的资料）
            topic:   报告主题（可选，自动从 content 推断）
            context: 额外的上下文信息（可选）

        Returns:
            包含 "success" 和 "report" 字段的字典
        """
        context = context or {}
        sources_summary = context.get("sources_summary", "知识库检索 / 用户提供")
        source_count = context.get("source_count", 0)

        # 如果未提供 topic，自动提取
        if not topic:
            topic = self._infer_topic(content, report_type)

        # 选择模板
        template = self._get_template(report_type)

        # 构造各章节 —— 优先用 LLM 生成，无模型时用规则抽取
        sections = self._build_sections(report_type, content, topic, context)

        # 填充模板
        filled = template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sources_summary=sources_summary,
            source_count=source_count,
            topic=topic,
            **sections,
        )

        return {
            "success": True,
            "report": filled,
            "report_type": report_type,
            "topic": topic,
            "sections": list(sections.keys()),
        }

    def _get_template(self, report_type: str) -> str:
        templates = {
            "deployment": self.DEPLOYMENT_TEMPLATE,
            "troubleshoot": self.TROUBLESHOOT_TEMPLATE,
            "meeting": self.MEETING_TEMPLATE,
        }
        return templates.get(report_type, self.MEETING_TEMPLATE)

    def _infer_topic(self, content: str, report_type: str) -> str:
        """从内容中提取标题/主题关键词"""
        # 优先匹配 Markdown 标题
        headings = re.findall(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
        if headings:
            return headings[0].strip()

        # 取前 60 字符
        plain = re.sub(r"[#*`>\-\[\]]", "", content[:120]).strip()
        return plain[:60] + ("…" if len(plain) > 60 else "")

    # ── 章节构建 ──────────────────────────────────────

    def _build_sections(
        self,
        report_type: str,
        content: str,
        topic: str,
        context: dict[str, Any],
    ) -> dict[str, str]:
        """为指定的报告类型填充所有章节"""
        if report_type == "deployment":
            return self._build_deployment_sections(content, topic, context)
        elif report_type == "troubleshoot":
            return self._build_troubleshoot_sections(content, topic, context)
        else:  # meeting
            return self._build_meeting_sections(content, topic, context)

    # ── 部署报告 ──────────────────────────────────────

    def _build_deployment_sections(self, content: str, topic: str, context: dict[str, Any]) -> dict[str, str]:
        # 优先使用 LLM，无则用规则裁剪
        if self.generator and len(content) > 200:
            return self._llm_deployment(content, topic)  # type: ignore[return-value]

        # 规则兜底
        return {
            "overview": self._para(content[:300], "本次部署涉及 " + topic),
            "environment": self._extract_env_info(content),
            "steps": self._numbered_list(content, "部署"),
            "config": self._extract_key_value_section(content, "配置"),
            "verification": self._para(content, "详见上方参考资料中的验证内容。"),
            "notes": self._extract_notes(content),
        }

    # ── 问题排查报告 ──────────────────────────────────

    def _build_troubleshoot_sections(self, content: str, topic: str, context: dict[str, Any]) -> dict[str, str]:
        if self.generator and len(content) > 200:
            return self._llm_troubleshoot(content, topic)  # type: ignore[return-value]

        return {
            "symptoms": self._extract_section_text(content, ["故障现象", "症状"]),
            "impact": self._extract_section_text(content, ["影响范围", "影响"]),
            "process": self._numbered_list(content, "排查"),
            "root_cause": self._extract_root_cause(content),
            "solution": self._extract_solution(content),
            "prevention": self._extract_prevention(content),
            "references": self._extract_section_text(content, ["参考", "相关文档"]),
        }

    # ── 会议纪要 ──────────────────────────────────────

    def _build_meeting_sections(self, content: str, topic: str, context: dict[str, Any]) -> dict[str, str]:
        if self.generator and len(content) > 200:
            return self._llm_meeting(content, topic)  # type: ignore[return-value]

        return {
            "overview": self._para(content[:200], f"会议主题：{topic}"),
            "discussion": self._numbered_list(content, "讨论"),
            "decisions": self._bullet_list(content, "决定|决议|确认"),
            "action_items": self._extract_action_items(content),
            "pending": self._bullet_list(content, "遗留|待议|后续"),
            "next_plan": self._para(content, "暂无明确的下次会议计划。"),
        }

    # ── LLM 辅助生成（各报告类型全量生成） ────────────

    def _llm_deployment(self, content: str, topic: str) -> dict[str, str]:
        prompt = f"""你是一个专业的报告生成助手。请根据以下资料内容和你的自身知识，生成一份**部署报告**的各章节内容。

## 报告主题
{topic}

## 参考资料
{content[:6000]}

## 要求
- **优先使用参考资料中的信息**，如果参考资料充分则严格基于资料
- **如果参考资料不足**，允许基于你的专业知识合理补充，但必须用（【自身知识】）标注补充内容
- 确保报告完整、专业、可操作
- 请按以下章节分别输出，每个章节用 ===章节名=== 分隔：

===部署概述===
（用 2-4 句话概括本次部署的背景、目标和范围）

===环境信息===
（列出部署环境：服务器、操作系统、依赖版本、网络拓扑等。参考资料不足时可根据常识补充通用环境要求）

===部署步骤===
（以数字列表形式列出部署的关键步骤。如果参考资料没有明确步骤，根据你的专业知识给出合理部署流程）

===配置说明===
（列出关键配置项及其说明。可根据专业知识给出通用配置建议）

===验证结果===
（描述如何验证部署成功，以及预期的验证结果）

===注意事项===
（列出部署过程中的注意事项和风险点）"""

        return self._llm_section_split(prompt)

    def _llm_troubleshoot(self, content: str, topic: str) -> dict[str, str]:
        prompt = f"""你是一个专业的报告生成助手。请根据以下资料内容和你的自身知识，生成一份**问题排查报告**的各章节内容。

## 报告主题
{topic}

## 参考资料
{content[:6000]}

## 要求
- **优先使用参考资料中的信息**，如果参考资料充分则严格基于资料
- **如果参考资料不足**，允许基于你的专业知识合理补充，但必须用（【自身知识】）标注
- 请按以下章节分别输出，每个章节用 ===章节名=== 分隔：

===故障现象===
（描述故障的具体表现。参考资料不足时可根据常识补充典型故障表现）

===影响范围===
（描述故障影响了哪些系统、服务或用户）

===排查过程===
（以数字列表形式列出排查的步骤和发现）

===根因分析===
（分析故障的根本原因。可根据专业知识给出常见根因分析）

===解决方案===
（列出解决该问题的具体方案和操作步骤）

===预防措施===
（列出如何避免类似问题再次发生的措施）

===相关参考===
（列出排查过程中参考的相关文档或资源）"""

        return self._llm_section_split(prompt)

    def _llm_meeting(self, content: str, topic: str) -> dict[str, str]:
        prompt = f"""你是一个专业的会议纪要生成助手。请根据以下会议记录内容，生成一份**会议纪要**的各章节内容。

## 会议主题
{topic}

## 会议记录
{content[:4000]}

## 要求
请按以下章节分别输出，每个章节用 ===章节名=== 分隔：

===会议概况===
（简要描述会议时间、参与方、会议目标）

===讨论内容===
（以数字列表形式列出会议的主要讨论点和关键发言）

===决议事项===
（列出会议达成的共识和决定）

===待办事项（Action Items）===
（列出具体的待办事项，每个事项包括：负责人、截止时间、具体任务）

===遗留问题===
（列出会议中未解决的问题和后续需要继续讨论的事项）

===下次会议计划===
（计划的下次会议时间、议题等。如果没有则写"未明确"）"""

        return self._llm_section_split(prompt)

    def _llm_section_split(self, prompt: str) -> dict[str, str]:
        """调用 LLM 并按 ===章节名=== 分割出各章节"""
        if not self.generator:
            return {"_fallback": "true"}

        try:
            raw = self.generator._call_openai(prompt)
        except Exception:
            return {"_fallback": "true"}

        # 按 ===xxx=== 分割
        sections: dict[str, str] = {}
        current_key = ""
        current_lines: list[str] = []

        for line in raw.split("\n"):
            m = re.match(r"===\s*(.+?)\s*===", line)
            if m:
                if current_key and current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_key and current_lines:
            sections[current_key] = "\n".join(current_lines).strip()

        # 规范化键名以匹配模板占位符
        key_map = {
            "部署概述": "overview",
            "环境信息": "environment",
            "部署步骤": "steps",
            "配置说明": "config",
            "验证结果": "verification",
            "注意事项": "notes",
            "故障现象": "symptoms",
            "影响范围": "impact",
            "排查过程": "process",
            "根因分析": "root_cause",
            "解决方案": "solution",
            "预防措施": "prevention",
            "相关参考": "references",
            "会议概况": "overview",
            "讨论内容": "discussion",
            "决议事项": "decisions",
            "待办事项（Action Items）": "action_items",
            "遗留问题": "pending",
            "下次会议计划": "next_plan",
        }

        mapped: dict[str, str] = {}
        for cn_key, en_key in key_map.items():
            if cn_key in sections:
                mapped[en_key] = sections[cn_key]

        return mapped

    # ── 规则辅助方法 ──────────────────────────────────

    @staticmethod
    def _para(content: str, fallback: str) -> str:
        """提取一段纯文本描述"""
        # 移除 markdown 标题标记和列表标记
        plain = re.sub(r"^#+\s*", "", content[:500], flags=re.MULTILINE)
        plain = re.sub(r"^[-*]\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"[#*`>]", "", plain).strip()
        if len(plain) < 20:
            return fallback
        # 取第一个完整的句子（以中文句号、感叹号、问号、换行符结尾）
        sentences = re.split(r"(?<=[。！？\n])", plain)
        result = sentences[0].strip() if sentences else fallback
        if len(result) < 20:
            return fallback
        return result

    @staticmethod
    def _numbered_list(content: str, keyword: str) -> str:
        """从内容中抽取含关键字的数字列表项（跳过标题行本身）

        匹配策略：找到含关键字的标题区段，然后提取其下的数字列表内容。
        """
        lines = content.split("\n")
        # 找到含关键字的区段标题行索引
        section_start = -1
        for i, l in enumerate(lines):
            if l.strip().startswith("#") and keyword in l:
                section_start = i
                break

        items = []
        if section_start >= 0:
            # 从标题下一行开始，提取数字列表项，直到遇到下一个标题
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if re.match(r"^\d+[.）\)]\s", stripped) and len(stripped) > 4:
                    clean = re.sub(r"^\d+[.）\)]\s*", "", stripped)
                    items.append(clean)

        # 如果没找到精确区段，则全文搜索
        if not items:
            for l in lines:
                stripped = l.strip()
                if re.match(r"^\d+[.）\)]\s", stripped) and len(stripped) > 4:
                    clean = re.sub(r"^\d+[.）\)]\s*", "", stripped)
                    items.append(clean)

        if not items:
            return f"请在资料中定位与「{keyword}」相关的内容。"
        result = []
        for i, item in enumerate(items[:8], 1):
            result.append(f"{i}. {item}")
        return "\n".join(result)

    @staticmethod
    def _bullet_list(content: str, keyword: str) -> str:
        """从内容中抽取含关键字的列表项（无序），支持 | 分隔的多关键词"""
        lines = content.split("\n")
        keywords = [k.strip() for k in keyword.split("|")]

        # 找含关键词的区段标题
        section_start = -1
        for i, l in enumerate(lines):
            stripped = l.strip()
            if stripped.startswith("#"):
                # 检查标题本身是否含关键词
                if any(k in stripped for k in keywords):
                    section_start = i
                    break

        items = []
        if section_start >= 0:
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    items.append(clean)
        else:
            # 全文搜索无序列表项
            for l in lines:
                stripped = l.strip()
                if stripped.startswith("#"):
                    continue
                if any(k in stripped for k in keywords) and len(stripped) > 6:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    items.append(clean)

        if not items:
            return "资料中未明确提及相关的内容。"
        result = []
        for item in items[:8]:
            result.append(f"- {item}")
        return "\n".join(result)

    @staticmethod
    def _extract_env_info(content: str) -> str:
        """尝试提取环境信息（section-aware）"""
        env_headers = ["环境要求", "环境信息", "系统要求", "环境", "硬件"]
        lines = content.split("\n")
        # 找最佳匹配的区段标题
        section_start = -1
        for kw in env_headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        env_lines = []
        if section_start >= 0:
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•\d+[.）\)]\s*", "", stripped)
                    clean = re.sub(r"^\d+[.）\)]\s*", "", clean)
                    env_lines.append(clean)

        if not env_lines:
            return "资料中未明确说明环境信息。"
        return "\n".join(f"- {l}" for l in env_lines[:6])

    @staticmethod
    def _extract_key_value_section(content: str, keyword: str) -> str:
        """提取 key-value 形式的配置段落"""
        lines = content.split("\n")
        # 查找含关键字的区段
        section_start = -1
        for i, l in enumerate(lines):
            if l.strip().startswith("#") and keyword in l:
                section_start = i
                break

        items = []
        if section_start >= 0:
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if ":" in stripped or "=" in stripped:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    items.append(clean)

        # 全文中搜索 key: value 或 key=value 行
        if not items:
            for l in lines:
                stripped = l.strip()
                if ("=" in stripped or ":" in stripped) and len(stripped) > 5:
                    if not stripped.startswith("#"):
                        clean = re.sub(r"^[-*•]\s*", "", stripped)
                        items.append(clean)

        if not items:
            return f"资料中未明确说明「{keyword}」相关内容。"
        return "\n".join(items[:8])

    @staticmethod
    def _extract_notes(content: str) -> str:
        """提取注意事项（section-aware）"""
        note_headers = ["注意事项", "注意", "警告", "风险提示"]
        lines = content.split("\n")

        section_start = -1
        for kw in note_headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    items.append(clean)
            if items:
                return "\n".join(f"- {l}" for l in items[:6])

        return "暂无特殊注意事项。"

    @staticmethod
    def _extract_root_cause(content: str) -> str:
        """提取根因分析（section-aware）"""
        headers = ["根因分析", "根因", "原因分析", "根本原因"]
        lines = content.split("\n")
        section_start = -1
        for kw in headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•\d+[.）\)]\s*", "", stripped)
                    clean = re.sub(r"^\d+[.）\)]\s*", "", clean)
                    items.append(clean)
            if items:
                return "\n".join(f"- {l}" for l in items[:4])

        return "资料中未明确说明根因，请结合排查过程自行判断。"

    @staticmethod
    def _extract_solution(content: str) -> str:
        """提取解决方案（section-aware）"""
        headers = ["解决方案", "解决", "修复方法", "处理"]
        lines = content.split("\n")
        section_start = -1
        for kw in headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 3:
                    # 去掉数字序号前缀（如 "1. "、"2、"）
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    clean = re.sub(r"^\d+[.、）\)]\s*", "", clean)
                    items.append(clean)
            if items:
                return "\n".join(f"- {l}" for l in items[:6])

        return "请参考上方参考资料中的相关解决方案。"

    @staticmethod
    def _extract_prevention(content: str) -> str:
        """提取预防措施（section-aware）"""
        headers = ["预防措施", "后续措施", "改进", "监控", "告警"]
        lines = content.split("\n")
        section_start = -1
        for kw in headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•\d+[.）\)]\s*", "", stripped)
                    clean = re.sub(r"^\d+[.）\)]\s*", "", clean)
                    items.append(clean)
            if items:
                return "\n".join(f"- {l}" for l in items[:4])

        return "资料中未明确说明预防措施。"

    @staticmethod
    def _extract_section_text(content: str, headers: list[str]) -> str:
        """从内容中提取指定标题下的文本正文（section-text）"""
        lines = content.split("\n")
        section_start = -1
        for kw in headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 3:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    clean = re.sub(r"^\d+[.、）\)]\s*", "", clean)
                    items.append(clean)
            if items:
                return "\n".join(items[:6])

        return "资料中未明确说明相关内容。"

    @staticmethod
    def _extract_action_items(content: str) -> str:
        """提取待办事项（section-aware）"""
        headers = ["待办事项", "待办", "todo", "action items", "后续工作"]
        lines = content.split("\n")
        section_start = -1
        for kw in headers:
            for i, l in enumerate(lines):
                if l.strip().startswith("#") and kw in l:
                    section_start = i
                    break
            if section_start >= 0:
                break

        if section_start >= 0:
            items = []
            for l in lines[section_start + 1 :]:
                stripped = l.strip()
                if stripped.startswith("#"):
                    break
                if len(stripped) > 4:
                    clean = re.sub(r"^[-*•]\s*", "", stripped)
                    items.append(clean)
            if items:
                table = ["| 待办事项 | 负责人 | 截止时间 |", "|----------|--------|----------|"]
                for item in items[:6]:
                    # 尝试从文本中提取"XXX负责/负责XXX"和"截止XX"
                    responsible = "—"
                    deadline = "—"
                    r_match = re.search(r"([^，,。]+)(?:负责|责任人)", item)
                    if r_match:
                        responsible = r_match.group(1).strip()
                    d_match = re.search(r"截止\s*(.+)", item)
                    if d_match:
                        deadline = d_match.group(1).strip()
                    # 从待办文本中去除 负责人/截止 描述
                    desc = re.sub(r"[,， ]*负责.*$", "", item)
                    desc = re.sub(r"[,， ]*截止.*$", "", desc)
                    table.append(f"| {desc.strip()} | {responsible} | {deadline} |")
                return "\n".join(table)

        return (
            "| 待办事项 | 负责人 | 截止时间 |\n|----------|--------|----------|\n| 资料中未明确列出待办事项 | — | — |"
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "tool_name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["deployment", "troubleshoot", "meeting"],
                        "description": "报告类型：deployment（部署报告）、troubleshoot（问题排查报告）、meeting（会议纪要）",
                    },
                    "content": {
                        "type": "string",
                        "description": "资料内容，通常是检索结果的文本拼接或用户直接提供的材料",
                    },
                    "topic": {
                        "type": "string",
                        "description": "报告主题，可选，留空则自动从资料中推断",
                    },
                },
                "required": ["report_type", "content"],
            },
        }

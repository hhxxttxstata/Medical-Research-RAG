"""
src/prompt_injection.py — 提示注入检测

规则引擎，不是 LLM-based detector。
扫描用户输入中的已知注入模式（角色扮演、越狱、系统提示覆盖）。
"""

import re

# ── 注入模式定义 ──────────────────────────────────────

# 每一条规则是一个 (pattern_name, compiled_regex) 元组
# 设计原则：
#   1. 只匹配英文/中文核心模式，避免对正常医学术语的过度拦截
#   2. 不区分大小写，匹配跨越空格和换行
#   3. 每个模式专注于一种注入技术，方便审计时精确定位

_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ── 系统提示覆盖 ──
    (
        "system_prompt_override",
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|directives|commands)", re.I | re.DOTALL),
    ),
    (
        "system_prompt_override",
        re.compile(r"forget\s+(all\s+)?(your\s+)?(role|persona|identity|system\s+prompt)", re.I | re.DOTALL),
    ),
    (
        "system_prompt_override",
        re.compile(r"you\s+are\s+(now|not\s+required\s+to)\b.*(doctor|assistant|expert|ai)", re.I | re.DOTALL),
    ),
    ("system_prompt_override", re.compile(r"new\s+(system\s+)?(prompt|instruction|rule)", re.I)),
    ("system_prompt_override", re.compile(r"\boverride\b", re.I)),
    (
        "system_prompt_override",
        re.compile(
            r"disregard|discard\s+(all\s+)?(rules|instructions|guidelines)",
            re.I,
        ),
    ),
    # ── 角色扮演 / 越狱 ──
    (
        "jailbreak",
        re.compile(
            r"dan|jailbreak|do\s+anything\s+now|no\s+(restrictions|limitations|boundaries)",
            re.I,
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"act\s+as\s+(if\s+you\s+are|though\s+you\s+are)\s+",
            re.I,
        ),
    ),
    ("jailbreak", re.compile(r"pretend\s+(to\s+be|that\s+you\s+are|you\s+are)", re.I)),
    # ── 医学问答越界 ──
    (
        "medical_boundary",
        re.compile(
            r"how\s+to\s+(make|create|synthesize|produce)\s+(a\s+)?(drug|medicine|poison)",
            re.I,
        ),
    ),
    (
        "medical_boundary",
        re.compile(
            r"ignore\s+(medical|clinical|safety|ethical)\s+(guidelines|protocol|rules)",
            re.I,
        ),
    ),
    (
        "medical_boundary",
        re.compile(
            r"give\s+me\s+(a\s+)?(prescription|diagnosis)\s+(for|without)\s+",
            re.I,
        ),
    ),
    # ── 提示泄露 ──
    (
        "prompt_leak",
        re.compile(
            r"(what\s+is|show|display|reveal|output)\s+(me\s+)?(your\s+)?(system\s+)?prompt",
            re.I,
        ),
    ),
    ("prompt_leak", re.compile(r"print\s+(your\s+)?(instructions|directives|guidelines|rules)", re.I)),
    ("prompt_leak", re.compile(r"(show|reveal|output)\s+(me\s+)?(the\s+)?(initial|first)\s+(prompt|message)", re.I)),
    # ── 中文注入模式 ──
    (
        "system_prompt_override",
        re.compile(
            r"(忽略|忘记|无视|遗忘|覆盖|跳过)(之前|先前|所有|以上|任何|一切|的|你)*?(指令|规则|要求|提示词|提示|系统提示|设定|角色)",
            re.I,
        ),
    ),
    (
        "system_prompt_override",
        re.compile(
            r"(你|您)(现在|从现在起|接下来|开始)?(要|必须|请)?(扮演|假装|作为|当|是)(一个|一名)?(医生|专家|助手|AI|人工智能)",
            re.I,
        ),
    ),
    (
        "system_prompt_override",
        re.compile(r"(新的|新)(系统提示|指令|规则|要求)", re.I),
    ),
    (
        "jailbreak",
        re.compile(r"(不受|没有|无需)(任何)?(限制|约束|边界)", re.I),
    ),
    (
        "jailbreak",
        re.compile(r"(做|执行|干)(任何|任意)(事情|操作|事)", re.I),
    ),
    (
        "medical_boundary",
        re.compile(r"(如何|怎么)(制造|合成|制作|配制)(药物|药品|毒药|毒品)", re.I),
    ),
    (
        "medical_boundary",
        re.compile(
            r"(忽略|无视)(医学|临床|安全|伦理)[^，。；\n]{0,6}?(准则|规范|规则|指南)",
            re.I,
        ),
    ),
    (
        "medical_boundary",
        re.compile(r"(给我|帮我)(开|写)(处方|药方)", re.I),
    ),
    (
        "prompt_leak",
        re.compile(r"(你的|您的)(系统)?(提示词|提示|指令|规则|要求|prompt)(是什么|有哪些|给我看|显示)", re.I),
    ),
    (
        "prompt_leak",
        re.compile(r"(输出|显示|泄露)(你的|您的)?(系统)?(提示词|提示|指令|规则|初始提示)", re.I),
    ),
]


def detect_injection(text: str) -> tuple[bool, str]:
    """检测输入文本中是否包含提示注入模式

    Args:
        text: 用户输入文本

    Returns:
        (is_injection: bool, reason: str)
        reason 在 is_injection=False 时为空字符串

    用法:
        is_injection, reason = detect_injection(user_question)
        if is_injection:
            return {"error": f"输入被拒绝: {reason}"}
    """
    if not text or not text.strip():
        return False, ""

    for category, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True, f"检测到 {category} 注入模式"

    return False, ""

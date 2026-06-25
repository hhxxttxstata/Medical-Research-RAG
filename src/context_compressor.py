"""
智能对话压缩模块 — 防止 Agent 多轮对话中 token 爆炸

核心策略：
  1. Tool Result 摘要：Observation > 2000 字符时用 LLM 压缩为 200 字精要
  2. 对话历史压缩：messages Token 数超预算时，淘汰旧轮 + 压缩为结构化摘要
  3. 无 LLM 时回退到提取式压缩（保留首尾关键句）

面试价值：
  - 展示对 LLM Context Window 管理的理解
  - 智能压缩 vs 暴力截断 — 用 LLM 做语义摘要而非丢弃信息
  - 与 BudgetTracker 的 Token 预算形成完整的流量控制系统
"""

from typing import Any

# ═══════════════════════════════════════════════════════════════
#  一、Token 估算工具
# ═══════════════════════════════════════════════════════════════


def estimate_tokens(text: str) -> int:
    """使用 tiktoken 估算 Token 数，失败时快速回退"""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        return ascii_chars // 4 + non_ascii // 2 + 1


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """估算 messages 列表的总 Token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += estimate_tokens(str(content))
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += estimate_tokens(tc["function"].get("arguments", ""))
    return total


# ═══════════════════════════════════════════════════════════════
#  二、Tool Result 智能摘要压缩器
# ═══════════════════════════════════════════════════════════════


class ToolResultCompressor:
    """工具执行结果（Observation）的智能压缩

    策略：
      - < 2000 字符：不压缩，直接透传
      - 2000-8000 字符：提取式压缩（保留首 500 + 尾 300 字符）
      - > 8000 字符：尝试 LLM 摘要（如有 generator），否则提取式
    """

    # 阈值
    PASS_THROUGH_LIMIT = 2000  # 低于此长度不压缩
    EXTRACTIVE_LIMIT = 8000  # 低于此长度用提取式，高于用 LLM
    EXTRACTIVE_HEAD = 600  # 提取式保留头部字符数
    EXTRACTIVE_TAIL = 300  # 提取式保留尾部字符数
    SUMMARY_MAX_CHARS = 400  # LLM 摘要最大字符数

    def __init__(self, generator=None):
        self.generator = generator

    def compress(self, observation: str, tool_name: str = "") -> str:
        """压缩一条工具输出的 Observation

        返回压缩后的文本。保证返回的文本不超过原始长度的 60%。
        """
        if len(observation) <= self.PASS_THROUGH_LIMIT:
            return observation

        # ── 优先尝试 LLM 摘要 ──
        if len(observation) > self.EXTRACTIVE_LIMIT and self.generator:
            summary = self._llm_summary(observation, tool_name)
            if summary and len(summary) < len(observation) * 0.8:
                return summary

        # ── 回退：提取式压缩 ──
        return self._extractive_compress(observation)

    def _llm_summary(self, observation: str, tool_name: str) -> str | None:
        """用 LLM 对工具输出做语义摘要"""
        try:
            compressed = self.generator.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是工具输出摘要助手。将工具的输出压缩为简洁的结构化摘要，"
                            f"不超过 {self.SUMMARY_MAX_CHARS} 字。保留所有关键数据、"
                            "数值、状态码、文件路径和错误信息。不要遗漏任何诊断结论。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"工具 [{tool_name}] 的输出如下。请压缩为 {self.SUMMARY_MAX_CHARS} 字以内的精要：\n\n{observation}",
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )
            if compressed and len(compressed) < len(observation) * 0.8:
                return compressed[: self.SUMMARY_MAX_CHARS]
        except Exception:
            pass
        return None

    def _extractive_compress(self, text: str) -> str:
        """提取式压缩：保留首部 + 尾部关键信息"""
        if len(text) <= self.PASS_THROUGH_LIMIT:
            return text

        head = text[: self.EXTRACTIVE_HEAD]
        tail = text[-self.EXTRACTIVE_TAIL :]

        # 头部去尾、尾部去头 — 避免从句子中间开始
        if head and head[-1] not in ("\n", "。", "，", ".", ",", "!", "?", "！", "？"):
            # 尝试回溯到最近的句号/换行
            cut = max(head.rfind("\n"), head.rfind("。"), head.rfind("."))
            if cut > self.EXTRACTIVE_HEAD // 2:
                head = head[: cut + 1]

        return f"{head}\n\n⋯[{len(text) - self.EXTRACTIVE_HEAD - self.EXTRACTIVE_TAIL} 字符已省略]⋯\n\n{tail}"


# ═══════════════════════════════════════════════════════════════
#  三、对话历史压缩器
# ═══════════════════════════════════════════════════════════════


class ConversationCompressor:
    """多轮对话消息列表的 Token 预算管理

    当 messages 列表的 Token 数超过阈值时，压缩旧轮次：
      1. 保留 system prompt + 最后 N 轮完整消息（PRESERVE_RECENT_TURNS）
      2. 中间旧轮次压缩为一个结构化摘要消息

    压缩后的消息列表结构：
      system
      user（摘要）← 旧轮次的压缩
      ... (保留的最近 N 轮完整消息)
    """

    # 压缩阈值（占总预算的百分比）
    TOKEN_BUDGET_RATIO = 0.7  # 达到总预算的 70% 时触发压缩
    PRESERVE_RECENT_TURNS = 3  # 保留最近 3 轮完整对话（比之前减少，更激进压缩）

    # 压缩后保留的 Token 目标（占预算的百分比）
    COMPRESS_TARGET_RATIO = 0.40  # 压缩后降到预算的 40%

    # 永远不会压缩的最小消息数（system + 至少 1 轮对话）
    MIN_PRESERVE_COUNT = 3  # system + 最后 1 轮 (user + assistant)

    def __init__(self, generator=None):
        self.generator = generator

    def maybe_compress(
        self,
        messages: list[dict[str, Any]],
        total_token_budget: int = 16384,
    ) -> list[dict[str, Any]]:
        """检查并压缩消息列表

        当估算 Token > budget * RATIO 时执行压缩。
        返回压缩后的消息列表（不修改原列表）。
        """
        current_tokens = estimate_messages_tokens(messages)
        threshold = int(total_token_budget * self.TOKEN_BUDGET_RATIO)

        if current_tokens <= threshold:
            return messages  # 未达阈值，不压缩

        return self._compress(messages, total_token_budget)

    def _compress(
        self,
        messages: list[dict[str, Any]],
        total_token_budget: int,
    ) -> list[dict[str, Any]]:
        """执行压缩：保留 system + 最近 N 轮 + 旧轮摘要"""
        if len(messages) <= self.MIN_PRESERVE_COUNT:
            return messages  # 太少，无法压缩

        # ── 分离 system prompt ──
        system_msg = messages[0] if messages[0].get("role") == "system" else None

        # ── 找出需要压缩的旧消息和保留的新消息 ──
        non_system = messages[1:] if system_msg else messages

        # 确保至少保留最后 1 轮完整对话（2 条消息），其余全压缩
        preserve_count = max(2, self.PRESERVE_RECENT_TURNS * 2)
        if len(non_system) <= preserve_count:
            return messages  # 不够一轮

        old_messages = non_system[:-preserve_count]
        recent_messages = non_system[-preserve_count:]

        # ── 构建压缩摘要 ──
        summary = self._build_summary(old_messages)
        summary_msg = {"role": "user", "content": summary}

        # ── 重组消息列表 ──
        result: list[dict[str, Any]] = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(recent_messages)

        return result

    def _build_summary(self, messages: list[dict[str, Any]]) -> str:
        """为旧消息构建压缩摘要"""
        # 提取关键信息
        actions = []
        results = []
        final_content = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    actions.append(tc["function"]["name"])

            if role == "tool" and content:
                # 截取 tool 结果的关键信息
                if len(content) > 200:
                    results.append(content[:200] + "…")
                else:
                    results.append(content)

            if role == "assistant" and content and not msg.get("tool_calls"):
                final_content = content[:500]

        parts = ["[对话历史摘要 — 以下为之前轮次的关键信息]"]

        if actions:
            parts.append(f"已调用的工具: {', '.join(actions)}")

        if results:
            # 最多保留 3 个工具结果的关键部分
            for r in results[-3:]:
                parts.append(f"- 工具结果: {r[:200]}")

        if final_content:
            parts.append(f"最近回答: {final_content[:300]}")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  四、统一的上下文压缩器（对外暴露入口）
# ═══════════════════════════════════════════════════════════════


class ContextCompressor:
    """统一的上下文压缩管理器

    组合 ToolResultCompressor 和 ConversationCompressor，
    提供一个一致的接口给 FunctionCallingLoop / ReActLoop 使用。

    用法:
        compressor = ContextCompressor(generator=generator, token_budget=16384)
        # 压缩单条工具输出
        obs = compressor.compress_tool_result(observation, tool_name)
        # 压缩消息列表
        messages = compressor.compress_conversation(messages)
    """

    def __init__(self, generator=None, token_budget: int = 16384):
        self.tool_compressor = ToolResultCompressor(generator=generator)
        self.conv_compressor = ConversationCompressor(generator=generator)
        self.token_budget = token_budget

    def compress_tool_result(self, observation: str, tool_name: str = "") -> str:
        """压缩单条工具输出的 Observation"""
        return self.tool_compressor.compress(observation, tool_name)

    def compress_conversation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """检查并压缩消息列表（需要时触发）"""
        return self.conv_compressor.maybe_compress(messages, self.token_budget)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return estimate_tokens(text)

    @staticmethod
    def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
        return estimate_messages_tokens(messages)

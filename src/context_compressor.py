"""
智能对话压缩 — 防止 Agent 多轮对话 Token 爆炸

策略：
  1. Tool Result 摘要：Observation > 2000 字符压缩
  2. 对话历史压缩：messages Token 超预算时淘汰旧轮 + 摘要
  3. 无 LLM 时提取式压缩（保留首尾）
"""

from typing import Any

# ── Token 估算 ──────────────────────────────────────


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars // 4 + (len(text) - ascii_chars) // 2 + 1


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(str(msg.get("content") or ""))
        for tc in msg.get("tool_calls") or []:
            total += estimate_tokens(tc["function"].get("arguments", ""))
    return total


# ── 统一的上下文压缩器 ──────────────────────────────


class ContextCompressor:
    """统一上下文压缩：工具结果压缩 + 对话历史压缩

    用法:
        cc = ContextCompressor(generator=gen, token_budget=16384)
        obs = cc.compress_tool_result(observation, "tool_name")
        messages = cc.compress_conversation(messages)
    """

    # Tool result 压缩阈值
    PASS_THROUGH_LIMIT = 2000
    LLM_SUMMARY_LIMIT = 8000
    EXTRACTIVE_HEAD = 600
    EXTRACTIVE_TAIL = 300
    SUMMARY_MAX_CHARS = 400

    # 对话压缩阈值
    TOKEN_BUDGET_RATIO = 0.7
    PRESERVE_RECENT_TURNS = 2

    def __init__(self, generator=None, token_budget: int = 16384):
        self.generator = generator
        self.token_budget = token_budget

    # ── Tool Result 压缩 ────────────────────────────

    def compress_tool_result(self, observation: str, tool_name: str = "") -> str:
        if len(observation) <= self.PASS_THROUGH_LIMIT:
            return observation

        if len(observation) > self.LLM_SUMMARY_LIMIT and self.generator:
            summary = self._llm_summary(observation, tool_name)
            if summary and len(summary) < len(observation) * 0.8:
                return summary

        return self._extractive_compress(observation)

    def _llm_summary(self, observation: str, tool_name: str) -> str | None:
        try:
            compressed = self.generator.chat(messages=[
                {"role": "system", "content": f"将工具输出压缩为{self.SUMMARY_MAX_CHARS}字以内的结构化摘要，保留关键数据。"},
                {"role": "user", "content": f"工具[{tool_name}]输出：\n\n{observation}"},
            ], temperature=0.0, max_tokens=512)
            if compressed and len(compressed) < len(observation) * 0.8:
                return compressed[:self.SUMMARY_MAX_CHARS]
        except Exception:
            pass
        return None

    def _extractive_compress(self, text: str) -> str:
        head = text[:self.EXTRACTIVE_HEAD]
        tail = text[-self.EXTRACTIVE_TAIL:]
        cut = max(head.rfind("\n"), head.rfind("。"), head.rfind("."))
        if cut > self.EXTRACTIVE_HEAD // 2:
            head = head[:cut + 1]
        return f"{head}\n\n⋯[{len(text) - self.EXTRACTIVE_HEAD - self.EXTRACTIVE_TAIL} 字符已省略]⋯\n\n{tail}"

    # ── 对话历史压缩 ────────────────────────────────

    def compress_conversation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Token 超预算时压缩旧轮次"""
        token_count = estimate_messages_tokens(messages)
        if token_count <= int(self.token_budget * self.TOKEN_BUDGET_RATIO):
            return messages

        if len(messages) <= 3:
            return messages

        system_msg = messages[0] if messages[0].get("role") == "system" else None
        non_system = messages[1:] if system_msg else messages
        preserve = max(2, self.PRESERVE_RECENT_TURNS * 2)
        if len(non_system) <= preserve:
            return messages

        old = non_system[:-preserve]

        # 构建摘要
        actions, results, final = [], [], ""
        for msg in old:
            if msg.get("tool_calls"):
                actions.extend(tc["function"]["name"] for tc in msg["tool_calls"])
            if msg.get("role") == "tool":
                c = (msg.get("content") or "")[:200]
                results.append(c + "…" if len(c) == 200 else c)
            if msg["role"] == "assistant" and msg.get("content") and not msg.get("tool_calls"):
                final = msg["content"][:500]

        summary_parts = ["[历史摘要]"]
        if actions:
            summary_parts.append(f"工具: {', '.join(set(actions))}")
        for r in results[-3:]:
            summary_parts.append(f"- {r[:200]}")
        if final:
            summary_parts.append(f"回答: {final[:300]}")

        result: list[dict[str, Any]] = []
        if system_msg:
            result.append(system_msg)
        result.append({"role": "user", "content": "\n".join(summary_parts)})
        result.extend(non_system[-preserve:])
        return result

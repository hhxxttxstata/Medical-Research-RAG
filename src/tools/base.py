"""
工具基类
定义 Agent 可调用工具的接口规范 + 工具调用策略 (Policy-as-Code)

Policy-as-Code 设计：
  - 每个工具声明自己的访问策略（ToolPolicy）
  - PolicyEnforcer 统一检查：auto→放行 / confirm→需用户确认 / manual→仅返回建议
  - Rate limiting 防止 Agent 失控调用
  - 审计理由要求跟踪意图

面试价值：
  - 展示对 Agent 安全边界的理解（在 Anthropic 2025 指南中是核心关注点）
  - Policy-as-Code 是零信任架构在 Agent 场景的实践
  - Human-in-the-Loop 三步分级：自动～建议+确认～仅人类可操作
"""

import time
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════
#  一、工具调用策略
# ═══════════════════════════════════════════════════════════════


@dataclass
class ToolPolicy:
    """工具调用策略

    Attributes:
        access_level: 访问级别
            - "auto" — Agent 可自动执行，无需确认
            - "confirm" — Agent 可调用，但需用户确认后实际执行
            - "manual" — Agent 仅返回建议，用户手动执行

        rate_limit: 每分钟最大调用次数（0 = 不限）

        allowed_roles: 允许调用此工具的角色列表（预留）

        require_reason: 调用时是否需要 Agent 附上理由（用于审计日志）
    """

    access_level: str = "auto"
    rate_limit: int = 0
    allowed_roles: list[str] = field(default_factory=lambda: ["admin", "user"])
    require_reason: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "access_level": self.access_level,
            "rate_limit": self.rate_limit,
            "allowed_roles": list(self.allowed_roles),
            "require_reason": self.require_reason,
        }


# ═══════════════════════════════════════════════════════════════
#  二、策略执行结果
# ═══════════════════════════════════════════════════════════════


@dataclass
class PolicyResult:
    """策略执行结果

    Attributes:
        allowed: True = 直接放行、等待确认也算放行（只是暂停）
        level: 实际匹配的 access_level
        reason: 拒绝/需要确认的原因
        needs_confirmation: True = 需要用户确认后才能继续
        retry_after: rate limit 场景下建议重试的等待秒数
    """

    allowed: bool = True
    level: str = "auto"
    reason: str = ""
    needs_confirmation: bool = False
    retry_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "level": self.level,
            "reason": self.reason,
            "needs_confirmation": self.needs_confirmation,
            "retry_after": self.retry_after,
        }


# ═══════════════════════════════════════════════════════════════
#  三、策略执行引擎
# ═══════════════════════════════════════════════════════════════


class PolicyEnforcer:
    """工具调用策略执行引擎

    负责：
      1. 按 access_level 判断是否允许/需要确认
      2. Rate limiting（滑动窗口）
      3. Reason 审计校验

    使用方式（单例，共享给所有 Agent 循环）:
        enforcer = PolicyEnforcer()
        policy = tool.policy
        result = enforcer.check("tool_name", policy, reason=...)
        if result.allowed and not result.needs_confirmation:
            enforcer.record_call("tool_name")
            # 执行工具...
        elif result.needs_confirmation:
            # 返回给前端等待用户确认
    """

    def __init__(self):
        # tool_name → [timestamp, ...]  滑动窗口调用记录
        self._call_history: dict[str, list[float]] = {}

    def check(
        self,
        tool_name: str,
        policy: ToolPolicy,
        reason: str = "",
    ) -> PolicyResult:
        """检查是否允许调用指定工具

        Args:
            tool_name: 工具名称
            policy: 工具策略定义
            reason: Agent 提供的调用理由（require_reason=True 时需要）

        Returns:
            PolicyResult
        """
        # ── 1. 角色检查（预留） ──
        # 当前所有角色通过，后续集成鉴权系统后在此处检查

        # ── 2. Reason 检查 ──
        if policy.require_reason and not reason.strip():
            return PolicyResult(
                allowed=False,
                level=policy.access_level,
                reason="该工具调用需要提供理由",
                needs_confirmation=False,
            )

        # ── 3. Rate limit 检查 ──
        if not self._check_rate_limit(tool_name, policy.rate_limit):
            retry_after = self._retry_after(tool_name, policy.rate_limit)
            return PolicyResult(
                allowed=False,
                level=policy.access_level,
                reason=f"工具调用频率超限，请在 {retry_after:.0f} 秒后重试",
                needs_confirmation=False,
                retry_after=retry_after,
            )

        # ── 4. Access level 判断 ──
        if policy.access_level == "auto":
            return PolicyResult(allowed=True, level="auto", reason="自动放行")

        if policy.access_level == "confirm":
            return PolicyResult(
                allowed=True,
                level="confirm",
                reason="该工具需用户确认后执行",
                needs_confirmation=True,
            )

        if policy.access_level == "manual":
            return PolicyResult(
                allowed=False,
                level="manual",
                reason="该工具需用户手动执行（Agent 仅提供建议）",
                needs_confirmation=False,
            )

        # 未知级别 → 安全拒绝
        return PolicyResult(
            allowed=False,
            level="unknown",
            reason=f"未知访问级别: {policy.access_level}",
        )

    def record_call(self, tool_name: str) -> None:
        """记录一次工具调用（用于 rate limit 计数）"""
        now = time.monotonic()
        if tool_name not in self._call_history:
            self._call_history[tool_name] = []
        # 清理 60 秒之前的记录
        cutoff = now - 60
        self._call_history[tool_name] = [t for t in self._call_history[tool_name] if t > cutoff]
        self._call_history[tool_name].append(now)

    def _check_rate_limit(self, tool_name: str, limit: int) -> bool:
        """检查是否超过限流阈值（滑动窗口，60 秒）"""
        if limit <= 0:
            return True  # 不限
        now = time.monotonic()
        cutoff = now - 60
        history = self._call_history.get(tool_name, [])
        # 只保留 60 秒内的记录
        recent = [t for t in history if t > cutoff]
        return len(recent) < limit

    def _retry_after(self, tool_name: str, limit: int) -> float:
        """建议重试等待秒数"""
        if limit <= 0:
            return 0.0
        history = self._call_history.get(tool_name, [])
        if len(history) < limit:
            return 0.0
        # 超 过限制后，最早的那次调用退休还需要多久
        cutoff = time.monotonic() - 60
        valid = [t for t in history if t > cutoff]
        if len(valid) >= limit:
            # 最旧的那个调用在 (valid[0] + 60) 时退休
            return max(0.0, valid[0] + 60 - time.monotonic())
        return 0.0

    def reset(self, tool_name: str | None = None) -> None:
        """重置调用记录（测试用）"""
        if tool_name:
            self._call_history.pop(tool_name, None)
        else:
            self._call_history.clear()


# ═══════════════════════════════════════════════════════════════
#  四、工具基类
# ═══════════════════════════════════════════════════════════════


class Tool:
    """Agent 工具基类

    所有工具必须继承此类并实现 run 方法。
    工具的描述和 schema 用于 Agent 理解其能力范围。
    工具的 policy 用于执行策略控制。
    """

    name: str = ""
    description: str = ""
    policy: ToolPolicy = field(default_factory=ToolPolicy)

    def run(self, **kwargs) -> dict[str, Any]:
        """执行工具逻辑

        Returns:
            包含执行结果的字典，必须包含 "success" 字段
        """
        raise NotImplementedError

    def get_schema(self) -> dict[str, Any]:
        """获取工具的 JSON Schema 描述（用于 Agent 意图匹配）"""
        schema = {
            "tool_name": self.name,
            "description": self.description,
            "parameters": self._get_parameter_schema(),
        }
        # 将 policy 附加在 schema 中供 Agent 展示
        schema["policy"] = self.policy.to_dict()
        return schema

    @property
    def openai_tool_schema(self) -> dict[str, Any]:
        """以 OpenAI Function Calling 格式返回工具定义

        将内部 get_schema() 格式转换为 OpenAI tools 参数格式。
        LLM 返回 tool_calls 时通过 name 字段匹配到对应工具。
        """
        schema = self.get_schema()
        params = schema.get("parameters", {})
        if "type" not in params:
            params["type"] = "object"
        return {
            "type": "function",
            "function": {
                "name": schema.get("tool_name", self.name),
                "description": schema.get("description", self.description),
                "parameters": params,
            },
        }

    @property
    def mcp_tool_schema(self) -> dict[str, Any]:
        """以 MCP (Model Context Protocol) 格式返回工具定义"""
        schema = self.get_schema()
        params = schema.get("parameters", {})
        if "type" not in params:
            params["type"] = "object"
        return {
            "name": schema.get("tool_name", self.name),
            "description": schema.get("description", self.description),
            "inputSchema": params,
        }

    def _get_parameter_schema(self) -> dict[str, Any]:
        """子类可重写以声明具体参数类型"""
        return {}

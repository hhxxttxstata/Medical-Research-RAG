"""
工具基类
定义 Agent 可调用工具的接口规范
"""

from typing import Any


class Tool:
    """Agent 工具基类

    所有工具必须继承此类并实现 run 方法。
    工具的描述和 schema 用于 Agent 理解其能力范围。
    """

    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> dict[str, Any]:
        """执行工具逻辑

        Returns:
            包含执行结果的字典，必须包含 "success" 字段
        """
        raise NotImplementedError

    def get_schema(self) -> dict[str, Any]:
        """获取工具的 JSON Schema 描述（用于 Agent 意图匹配）"""
        return {
            "tool_name": self.name,
            "description": self.description,
            "parameters": self._get_parameter_schema(),
        }

    @property
    def openai_tool_schema(self) -> dict[str, Any]:
        """以 OpenAI Function Calling 格式返回工具定义

        将内部 get_schema() 格式:
            {"tool_name": ..., "description": ..., "parameters": {...}}
        转换为 OpenAI tools 参数格式:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

        LLM 返回 tool_calls 时通过 name 字段匹配到对应工具。
        """
        schema = self.get_schema()
        # 确保 param schema 包含 type="object"（OpenAI API 要求）
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
        """以 MCP (Model Context Protocol) 格式返回工具定义

        MCP Tool 采用直接 JSON Schema 描述参数，无外层嵌套：
            {
                "name": "...",
                "description": "...",
                "inputSchema": {"type": "object", "properties": {...}, ...}
            }

        这使该工具可直接被 MCP Host（如 Claude Desktop）识别和调用。
        """
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

"""
Agent 模块 — LLM 驱动意图分类 + ReAct 循环

核心流程：
  用户请求 → LLMIntentClassifier(LLM 分类，规则兜底)
             └→ ReActLoop(Thought → Action → Observation → ... → Final Answer)
                  └→ 调用工具
                  └→ 返回最终结果

面试价值：
  - ReAct 循环是 Agent 面试第一题，自实现展示深入理解
  - LLM 替代规则做意图分类，处理自然语言变体
  - 完整的 Max Steps 终止条件 + 死循环检测
  - 规则兜底保证系统在任何情况下都不崩溃
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import context_compressor
from .memory import MemoryManager
from .monitoring.metrics import record_agent_budget_reason, record_agent_steps, record_agent_token_usage, record_request

# ── 可观测性 ────────────────────────────────────────
from .monitoring.tracing import get_tracer
from .tools.base import PolicyEnforcer, PolicyResult, Tool
from .tools.diagnosis_tool import DiagnosisTool
from .tools.report_generator import ReportGenerator

# ════════════════════════════════════════════════════════════════
#  零、Bounded Agent Loop — 预算追踪
# ════════════════════════════════════════════════════════════════


@dataclass
class AgentHarnessConfig:
    """Agent 运行时配置 — 预算 + 安全 + 压缩

    一次性注入所有 Harness 层参数，避免分散到各 __init__ 中。
    """

    max_steps: int = 10
    """最大推理步数"""
    max_tokens_total: int = 16384
    """会话累计 Token 上限"""
    max_wall_clock_sec: float = 120.0
    """墙钟时间上限（秒）"""
    max_tool_calls: int = 20
    """工具调用总次数上限"""
    dead_loop_threshold: int = 3
    """连续重复操作触发死循环检测的阈值"""

    # ── 上下文压缩 ──
    enable_compression: bool = True

    # ── 安全策略 ──
    confirm_handler: Callable | None = None
    """Human-in-the-Loop 确认回调"""


class BudgetTracker:
    """Agent 预算跟踪器（步数 / Token / 墙钟 / 工具调用次数）

    每步推理前后调用 check_* 方法，返回 None = 正常, 字符串 = 终止原因。
    """

    def __init__(self, config: AgentHarnessConfig | None = None):
        self.cfg = config or AgentHarnessConfig()
        self.reset()

    def reset(self):
        self.step_count: int = 0
        self.token_count: int = 0
        self.tool_call_count: int = 0
        self._start_time: float = time.monotonic()

    def record_step(self, tokens: int = 0) -> str | None:
        self.step_count += 1
        self.token_count += tokens
        if self.step_count > self.cfg.max_steps:
            record_agent_budget_reason("步数超限")
            return "达到最大推理步数"
        if self.token_count > self.cfg.max_tokens_total:
            record_agent_budget_reason("Token预算超限")
            return f"会话 Token 超限 ({self.token_count}>{self.cfg.max_tokens_total})"
        return None

    def record_tool_call(self) -> str | None:
        self.tool_call_count += 1
        if self.tool_call_count > self.cfg.max_tool_calls:
            record_agent_budget_reason("工具调用次数超限")
            return f"工具调用超限 ({self.tool_call_count}>{self.cfg.max_tool_calls})"
        return None

    def check_wall_clock(self) -> str | None:
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.cfg.max_wall_clock_sec:
            record_agent_budget_reason("超时")
            return f"执行超时 ({elapsed:.1f}s>{self.cfg.max_wall_clock_sec}s)"
        return None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        try:
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars // 4 + (len(text) - ascii_chars) // 2 + 1

    def finalize(self):
        record_agent_token_usage(self.token_count)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time


# ════════════════════════════════════════════════════════════════
#  一、关键词意图分析器（保留做冷启动兜底）
# ════════════════════════════════════════════════════════════════


class IntentAnalyzer:
    """意图分析器（规则版）

    关键词匹配 + 模式识别，用于 LLM 不可用时的冷启动兜底。
    与 LLMIntentClassifier 共存，作为 fallback 链路。
    """

    # ── 报告意图的关键词映射 ──────────────────────────
    REPORT_PATTERNS = {
        "deployment": [
            "部署",
            "上线",
            "发布",
            "环境搭建",
            "deployment",
            "deploy report",
        ],
        "troubleshoot": [
            "排查",
            "故障",
            "解决",
            "是什么原因",
            "为什么报错",
            "根因",
            "troubleshoot",
            "incident report",
            "错误",
            "报错",
            "失败",
            "异常",
        ],
        "meeting": [
            "会议",
            "纪要",
            "meeting minutes",
        ],
    }

    # ── 肺栓塞诊断意图的关键词映射 ──────────────────────
    DIAGNOSIS_PATTERNS = [
        "肺栓塞诊断",
        "诊断肺栓塞",
        "肺栓塞预测",
        "预测肺栓塞",
        "栓塞检测",
        "栓塞预测",
        "栓塞识别",
        "pe诊断",
        "pe检测",
        "pe预测",
        "影像诊断",
        "分析ct",
        "分析影像",
        "读片",
        "阅片",
        "预测风险",
        "风险评估",
        "是不是肺栓塞",
        "是否肺栓塞",
        "有没有肺栓塞",
        "diagnose",
        "pe diagnosis",
        "embolism detection",
        "ctpa analysis",
        "诊断一下",
        "检测一下",
        "预测一下",
        "读一下",
    ]

    TOOL_MAP: dict[str, str] = {
        "deployment": "generate_report",
        "troubleshoot": "generate_report",
        "meeting": "generate_report",
        "pe_diagnosis": "diagnose_pulmonary_embolism",
    }

    @classmethod
    def analyze(cls, query: str) -> dict[str, Any]:
        """分析用户问题，返回意图识别结果"""
        query_lower = query.lower().strip()

        # ── 1. 检查是否肺栓塞诊断意图 ──
        diagnosis_score = cls._match_patterns(query_lower, cls.DIAGNOSIS_PATTERNS)
        if diagnosis_score >= 0.3:
            return {
                "intent": "pe_diagnosis",
                "tool_name": cls.TOOL_MAP.get("pe_diagnosis"),
                "report_type": None,
                "confidence": round(min(diagnosis_score, 1.0), 2),
            }

        # ── 2. 检查是否报告生成意图 ──
        best_match = None
        best_score = 0.0

        for report_type, patterns in cls.REPORT_PATTERNS.items():
            score = cls._match_patterns(query_lower, patterns)
            if score > best_score:
                best_score = score
                best_match = report_type

        if best_match and best_score >= 0.3:
            return {
                "intent": "report_generate",
                "tool_name": cls.TOOL_MAP.get(best_match),
                "report_type": best_match,
                "confidence": round(best_score, 2),
            }

        # ── 3. 默认走常规 RAG 问答 ──
        return {
            "intent": "normal_query",
            "tool_name": None,
            "report_type": None,
            "confidence": 1.0,
        }

    @classmethod
    def _match_patterns(cls, text: str, patterns: list[str]) -> float:
        if not patterns:
            return 0.0
        matches = 0
        for pat in patterns:
            if pat.lower() in text.lower():
                matches += 1
        score = matches * 0.3
        return min(score, 1.0)


# ════════════════════════════════════════════════════════════════
#  二、LLM 驱动意图分类器
# ════════════════════════════════════════════════════════════════

_INTENT_CLASSIFICATION_SYSTEM_PROMPT = """\
你是一个意图识别助手。请判断用户问题的意图，从以下分类中选择：

## 意图分类

### 1. pe_diagnosis（肺栓塞影像诊断）
用户想要分析 CT/CTPA 影像，诊断是否存在肺栓塞。
这类请求通常涉及：影像分析、读片、诊断、预测。
示例："帮我看看这个CT片子有没有问题"、"诊断一下是不是肺栓塞"、"分析这个影像"

### 2. report_generate（生成结构化报告）
用户需要生成部署文档、问题排查报告、会议纪要等结构化文档。
子类型：
- deployment: 部署/发布/环境搭建相关
- troubleshoot: 问题排查/故障分析/错误排查相关
- meeting: 会议纪要/讨论记录相关
示例："生成一份部署报告"、"排查一下为什么报错"、"整理会议纪要"

### 3. normal_query（常规问答）
用户的询问不需要调用专用工具，属于知识性问答。
示例："什么是肺栓塞"、"RAG有哪些应用"、"怎么优化检索效果"

## 输出格式
请严格按以下 JSON 格式输出，不要包含其他内容：
{"intent": "pe_diagnosis | report_generate | normal_query", "report_type": "deployment | troubleshoot | meeting | null", "reasoning": "简短判断理由"}

注意：
- report_type 仅在 intent 为 report_generate 时需要指定
- 如果不确定，优先选择 normal_query"""


class LLMIntentClassifier:
    """LLM 驱动的意图分类器

    调用大模型判断用户意图，得到结构化分类结果。
    如果 LLM 调用失败或返回格式异常，自动回退到 IntentAnalyzer（规则兜底）。

    面试价值：
      - LLM 分类能理解自然语言变体（"读一下这个片子" vs "诊断肺栓塞"）
      - 保留规则兜底确保系统在任何情况下都不崩溃
      - 纯文本 prompt + JSON 输出，不依赖专用分类 API
    """

    def __init__(self, generator=None):
        self.generator = generator

    def classify(self, query: str) -> dict[str, Any]:
        """分类用户意图

        返回兼容 IntentAnalyzer.analyze() 格式的字典：
            intent, tool_name, report_type, confidence, llm_classified
        """
        if self.generator is None:
            return self._rule_fallback(query)

        try:
            result = self._llm_classify(query)
            if result is not None:
                return result
        except Exception:
            pass

        return self._rule_fallback(query)

    def _llm_classify(self, query: str) -> dict[str, Any] | None:
        """调用 LLM 进行意图分类，返回 None 表示解析失败需回退"""
        response = self.generator.chat(
            messages=[
                {"role": "system", "content": _INTENT_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,  # 分类任务需要确定性
            max_tokens=256,
        )

        parsed = self._parse_json(response)
        if parsed is None:
            return None

        intent = parsed.get("intent", "normal_query")
        if intent not in ("pe_diagnosis", "report_generate", "normal_query"):
            return None

        # 构建兼容返回格式
        tool_name = None
        report_type = None
        if intent == "pe_diagnosis":
            tool_name = "diagnose_pulmonary_embolism"
        elif intent == "report_generate":
            tool_name = "generate_report"
            rt = parsed.get("report_type", "deployment")
            report_type = rt if rt in ("deployment", "troubleshoot", "meeting") else "deployment"

        return {
            "intent": intent,
            "tool_name": tool_name,
            "report_type": report_type,
            "confidence": 1.0,
            "llm_classified": True,
        }

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """从 LLM 输出中健壮地提取 JSON"""
        text = text.strip()
        # 去除 markdown 代码块标记
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 {...} 块
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _rule_fallback(query: str) -> dict[str, Any]:
        """回退到规则分类"""
        result = IntentAnalyzer.analyze(query)
        result["llm_classified"] = False
        return result


# ════════════════════════════════════════════════════════════════
#  三、Agent Harness — 统一运行时配置 + 公共循环基类
# ════════════════════════════════════════════════════════════════


@dataclass
class AgentHarnessConfig:
    """Agent 运行时统一配置

    一次性注入 Budget / Compressor / Policy / HITL 等 Harness 层参数，
    避免分散到 FunctionCallingLoop 和 ReActLoop 的 __init__ 中。

    面试价值：
      - 将所有 Harness 运行时配置集中管理 → 配置化而非硬编码
      - 面试官看一眼就知道你的 Agent 有哪些"安全气囊"
      - 新增策略时只需改一个 dataclass，不改循环代码
    """

    max_steps: int = 10
    """最大推理步数"""
    max_tokens_total: int = 16384
    """会话累计 Token 上限"""
    max_wall_clock_sec: float = 120.0
    """墙钟时间上限（秒）"""
    max_tool_calls: int = 20
    """工具调用总次数上限"""
    dead_loop_threshold: int = 3
    """连续重复操作触发死循环检测的阈值"""

    # ── 上下文压缩 ──
    enable_compression: bool = True
    """是否启用对话上下文压缩"""

    # ── 安全策略 ──
    confirm_handler: Callable | None = None
    """Human-in-the-Loop 确认回调（CLI 模式传入）"""

class AgentLoopBase:
    """Agent 消息循环的公共基类 — Harness 层

    统一 FunctionCallingLoop 和 ReActLoop 的：
      - 初始化（BudgetTracker / Compressor / PolicyEnforcer）
      - 预算检查（墙钟检查、步数+Token、工具调用次数）
      - 工具执行（_execute_action / _execute_retrieve / Policy 检查）
      - 结果封装（_build_result — 统一返回格式）

    子类只需实现：
      - run()          — 具体的循环逻辑（FC vs ReAct 文本协议）
      - _llm_step()    — 单步 LLM 推理（chat_with_tools vs chat）
      - _handle_llm_response() — 处理 LLM 响应（tool_calls vs 文本解析）
    """

    DEAD_LOOP_THRESHOLD = 3

    def __init__(
        self,
        tools: dict[str, "Tool"],
        harness_config: AgentHarnessConfig | None = None,
        generator=None,
        rag_pipeline=None,
    ):
        self.tools = tools
        self.generator = generator
        self.rag_pipeline = rag_pipeline
        self._config = harness_config or AgentHarnessConfig()
        self.max_steps = self._config.max_steps

        # ── 预算控制 ──
        self._budget_tracker: BudgetTracker | None = None

        # ── 安全策略 ──
        self._policy_enforcer = PolicyEnforcer()
        self.confirm_handler = self._config.confirm_handler

        # ── 上下文压缩 ──
        if self._config.enable_compression and generator:
            self._compressor = context_compressor.ContextCompressor(
                generator=generator,
                token_budget=self._config.max_tokens_total,
            )
        else:
            self._compressor = context_compressor.ContextCompressor(
                generator=None,
                token_budget=self._config.max_tokens_total,
            )

    # ════════════════════════════════════════════════════════
    #  —— 子类必须实现 ——
    # ════════════════════════════════════════════════════════

    def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError

    # ════════════════════════════════════════════════════════
    #  —— 预算检查（公共） ——
    # ════════════════════════════════════════════════════════

    def _init_budget(self) -> None:
        """在每个 run() 开始时初始化 BudgetTracker"""
        self._budget_tracker = BudgetTracker(self._config)

    def _check_wall_clock(self) -> str | None:
        """墙钟检查 → 返回终止原因或 None"""
        if self._budget_tracker:
            return self._budget_tracker.check_wall_clock()
        return None

    def _record_step(self, tokens: int) -> str | None:
        """步数 + Token 预算检查"""
        if self._budget_tracker:
            return self._budget_tracker.record_step(tokens)
        return None

    def _record_tool_calls(self, count: int = 1) -> str | None:
        """工具调用次数检查"""
        if self._budget_tracker:
            for _ in range(count):
                reason = self._budget_tracker.record_tool_call()
                if reason:
                    return reason
        return None

    def _finalize_budget(self) -> None:
        if self._budget_tracker:
            self._budget_tracker.finalize()

    @staticmethod
    def _build_user_message(query: str, context: dict[str, Any]) -> str:
        """构建用户消息（公共）"""
        extra = ""
        if context.get("file_path"):
            extra += f"\n用户已提供文件路径：{context['file_path']}"
        if context.get("report_type"):
            extra += f"\n用户指定报告类型：{context['report_type']}"
        return f"用户问题：{query}{extra}"

    # ════════════════════════════════════════════════════════
    #  —— Channel 层：消息生命周期，可被 Tracing 包裹 ——
    # ════════════════════════════════════════════════════════

    def _compress_observation(self, observation: str, tool_name: str = "") -> str:
        """压缩工具输出"""
        return self._compressor.compress_tool_result(observation, tool_name)

    def _compress_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """压缩对话历史"""
        return self._compressor.compress_conversation(messages)

    # ════════════════════════════════════════════════════════
    #  —— 工具执行（公共） ——
    # ════════════════════════════════════════════════════════

    def _execute_action(
        self,
        action: str,
        action_input: Any,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """执行工具调用或内置命令，返回 Observation 文本"""
        # ── 内置命令：知识库检索 ──
        if action == "retrieve":
            return self._execute_retrieve(action_input, query)

        # ── 注册工具 ──
        if action in self.tools:
            tool = self.tools[action]
            params = action_input if isinstance(action_input, dict) else {}
            for key in ("file_path",):
                if key not in params and context.get(key):
                    params[key] = context[key]
            try:
                result = tool.run(**params)
                return json.dumps(result, ensure_ascii=False, default=str)[:12000]
            except Exception as e:
                return f"工具调用失败：{e}"

        return f"未知命令：{action}。可用工具：{list(self.tools.keys())}，内置命令：['retrieve']"

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        query: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """并行执行多个 tool_calls（Function Calling 路径用）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def execute_one(tc: dict[str, Any]) -> dict[str, Any]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}

            for key in ("file_path",):
                if key not in args and context.get(key):
                    args[key] = context[key]

            if name == "retrieve":
                obs = self._execute_retrieve(args, query)
                return {"observation": obs, "structured": None}

            if name in self.tools:
                tool = self.tools[name]
                try:
                    result = tool.run(**args)
                    obs = json.dumps(result, ensure_ascii=False, default=str)[:12000]
                    return {"observation": obs, "structured": result}
                except Exception as e:
                    return {"observation": f"工具调用失败：{e}", "structured": None}

            return {"observation": f"未知工具：{name}", "structured": None}

        n = len(tool_calls)
        results: list[dict[str, Any] | None] = [None] * n

        with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
            future_map = {executor.submit(execute_one, tc): i for i, tc in enumerate(tool_calls)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"observation": f"并行执行错误：{e}", "structured": None}

        return [r if r is not None else {"observation": "工具执行未返回结果", "structured": None} for r in results]

    def _execute_retrieve(self, action_input: Any, default_query: str) -> str:
        """执行知识库检索（公共）"""
        if not self.rag_pipeline or not self.rag_pipeline.retriever:
            return "检索功能不可用（未连接知识库）"

        q = default_query
        k = 5
        if isinstance(action_input, dict):
            q = action_input.get("query", default_query)
            k = action_input.get("top_k", 5)

        try:
            chunks = self.rag_pipeline.retriever.retrieve(q, top_k=k)
            if not chunks:
                return "知识库中未找到相关信息。"
            parts = []
            for i, c in enumerate(chunks[:5], 1):
                text = c["text"][:300]
                src = c["metadata"].get("filename", "未知")
                parts.append(f"[{i}]（来源：{src}）{text}")
            return "\n\n".join(parts)
        except Exception as e:
            return f"检索出错：{e}"

    # ════════════════════════════════════════════════════════
    #  —— Policy 检查（公共） ——
    # ════════════════════════════════════════════════════════

    def _check_tool_policy(self, tool_name: str, reason: str = "", session_id: str = "") -> "PolicyResult | None":
        """检查工具调用策略

        Args:
            tool_name: 工具名称
            reason:    调用理由
            session_id: 会话 ID（用于按会话隔离 rate limit）

        Returns:
            PolicyResult | None — None 表示 auto 放行，否则包含策略判定结果
        """
        if tool_name == "retrieve":
            return None

        tool = self.tools.get(tool_name)
        if tool is None:
            return None

        policy = getattr(tool, "policy", None)
        if policy is None:
            return None

        result = self._policy_enforcer.check(tool_name, policy, reason, session_id)
        if result.level == "auto" and result.allowed:
            return None

        return result

    @staticmethod
    def _extract_reason(tc: dict[str, Any]) -> str:
        """从 tool_call 的 arguments 中提取 'reason' 参数"""
        try:
            args = json.loads(tc["function"]["arguments"])
            if isinstance(args, dict):
                return args.get("reason", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return ""

    # ════════════════════════════════════════════════════════
    #  —— 结果封装（公共） ——
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _build_result(
        success: bool,
        result: Any,
        steps: int,
        reason: str,
        trace: list[dict[str, Any]],
        tool_result: Any = None,
    ) -> dict[str, Any]:
        """统一返回格式"""
        return {
            "success": success,
            "result": result,
            "tool_result": tool_result,
            "steps": steps,
            "termination_reason": reason,
            "trace": trace,
        }


# ════════════════════════════════════════════════════════════════
#  四、Function Calling 循环（原生 OpenAI tools 支持）
# ════════════════════════════════════════════════════════════════

_FUNCTION_CALLING_SYSTEM_PROMPT = """\
你是一个智能助手，通过 Function Calling 方式工作。你可以调用的工具已通过函数列表提供。

## 可用命令
除了注册的工具，你还可以使用内置的 **retrieve** 命令从知识库检索信息。

## 工作方式
1. 分析用户问题，判断是否需要调用工具
2. 如果需要：返回 tool_calls（可以一次调用多个并行工具）
3. 工具执行后你会收到结果，基于结果继续推理
4. 如果不需要工具或信息已足够：直接输出最终回答文本

## 规则
- 最多 {max_steps} 轮工具调用
- 如果同一操作重复多次得到相同结果，请停止并基于已有信息回答
- 如果无法完成任务，诚实说明"""


class FunctionCallingLoop(AgentLoopBase):
    """基于原生 OpenAI Function Calling 的 ReAct 循环

    相比 ReActLoop（基于文本 Action/Action Input 正则解析）：
      - 使用 API 原生的 tools 参数，LLM 返回结构化 tool_calls
      - 支持并行工具调用（一次返回多个 tool_calls）
      - 不需要用正则解析，格式由 API 保证
      - API 不支持 Function Calling 时自动降级到 ReActLoop

    终止条件（与 ReActLoop 一致）：
      1. LLM 返回纯文本（无 tool_calls）→ 最终回答
      2. 达到 MAX_STEPS → 超时终止
      3. 死循环检测 → 连续 N 次相同操作
      4. BudgetTracker → Token/墙钟/工具调用次数超限（新增）
    """

    def __init__(
        self,
        tools: dict[str, "Tool"],
        generator=None,
        rag_pipeline=None,
        harness_config: AgentHarnessConfig | None = None,
    ):
        super().__init__(
            tools=tools,
            harness_config=harness_config,
            generator=generator,
            rag_pipeline=rag_pipeline,
        )
        self._fc_supported: bool | None = None  # 缓存探测结果

    # ── 公共入口 ──────────────────────────────────────

    def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """执行 Function Calling 循环"""
        context = context or {}
        openai_tools = self._build_openai_tools()

        # ── 探测 Function Calling 支持 ──
        if not self._check_fc_support(openai_tools):
            fallback = ReActLoop(
                tools=self.tools,
                generator=self.generator,
                rag_pipeline=self.rag_pipeline,
                harness_config=self._config,
            )
            return fallback.run(query, context=context, memory_context=memory_context, session_id=session_id)

        # ── 初始化 BudgetTracker ──
        self._init_budget()

        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
            {"role": "user", "content": self._build_user_message(query, context)},
        ]

        trace: list[dict[str, Any]] = []
        dead_loop_window: list[tuple[str, str]] = []
        last_tool_result: dict[str, Any] | None = None

        for step in range(1, self.max_steps + 1):
            # ── 预算：墙钟 ──
            reason = self._check_wall_clock()
            if reason:
                self._finalize_budget()
                return self._build_result(False, reason, step, reason, trace, last_tool_result)

            # ── 1. LLM（带 tools） ──
            response = self.generator.chat_with_tools(
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2048,
                parallel_tool_calls=True,
            )

            # ── 显式降级检查：API 不可用 → 降级到 ReAct 文本协议 ──
            if response.is_degraded:
                fallback = ReActLoop(
                    tools=self.tools,
                    generator=self.generator,
                    rag_pipeline=self.rag_pipeline,
                    harness_config=self._config,
                )
                return fallback.run(query, context=context, memory_context=memory_context, session_id=session_id)

            # ── 预算：步数 + Token ──
            response_text = response.content or ""
            step_tokens = BudgetTracker.estimate_tokens(response_text)
            reason = self._record_step(step_tokens)
            if reason:
                self._finalize_budget()
                return self._build_result(False, reason, step, reason, trace, last_tool_result)

            # ── 2. 追加 assistant ──
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            trace.append(
                {
                    "step": step,
                    "content": response.content,
                    "tool_calls": [
                        {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}
                        for tc in response.tool_calls
                    ]
                    if response.tool_calls
                    else [],
                }
            )

            # ── 3. 纯文本 → 最终答案 ──
            if not response.tool_calls and response.content:
                self._finalize_budget()
                return self._build_result(True, response.content, step, "正常完成", trace, last_tool_result)

            # ── 4. 执行 tool_calls ──
            if response.tool_calls:
                # ── 预算：工具调用次数 ──
                reason = self._record_tool_calls(len(response.tool_calls))
                if reason:
                    self._finalize_budget()
                    return self._build_result(False, reason, step, reason, trace, last_tool_result)

                # ── Policy 检查 ──
                pending_confirmation = False
                policy_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for tc in response.tool_calls:
                    name = tc["function"]["name"]
                    reason_arg = self._extract_reason(tc)
                    policy_result = self._check_tool_policy(name, reason_arg, session_id)
                    if policy_result and (policy_result.needs_confirmation or not policy_result.allowed):
                        pending_confirmation = True
                    policy_results.append((tc, policy_result.to_dict()))

                if pending_confirmation:
                    if self.confirm_handler:
                        confirmed = self.confirm_handler(policy_results, session_id)
                        if not confirmed:
                            rejection = (
                                "用户已拒绝执行该操作。请向用户说明你为何需要调用此工具，"
                                "或基于已有信息回答。如果无法完成请求，请诚实告知用户。"
                            )
                            for tc in response.tool_calls:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc["id"],
                                        "content": rejection,
                                    }
                                )
                            last_tool_result = {"success": False, "error": "用户拒绝"}
                            continue
                    else:
                        self._finalize_budget()
                        return self._build_result(
                            False,
                            {"pending_calls": [pr for _, pr in policy_results if pr["level"] != "auto"]},
                            step,
                            "需要用户确认",
                            trace,
                            last_tool_result,
                        )

                tool_results = self._execute_tool_calls(response.tool_calls, query, context)

                for tc in response.tool_calls:
                    self._policy_enforcer.record_call(tc["function"]["name"], session_id)

                # ── 预算：工具结果 Token ──
                obs_text = " ".join(tr.get("observation", "") for tr in tool_results)
                obs_tokens = BudgetTracker.estimate_tokens(obs_text)
                reason = self._record_step(obs_tokens)
                if reason:
                    self._finalize_budget()
                    return self._build_result(False, reason, step, reason, trace, last_tool_result)

                # ── 追加 tool results（压缩） ──
                for tc, tr in zip(response.tool_calls, tool_results):
                    obs = self._compress_observation(tr["observation"], tc["function"]["name"])
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})

                for tr in tool_results:
                    if tr.get("structured"):
                        last_tool_result = tr["structured"]

                # ── 5. 死循环检测 ──
                sig_parts = [tc["function"]["name"] for tc in response.tool_calls]
                obs_parts = [tr["observation"][:300] for tr in tool_results]
                dead_loop_window.append(("|".join(sig_parts), "|".join(obs_parts)))
                if len(dead_loop_window) >= self._config.dead_loop_threshold:
                    recent = dead_loop_window[-self._config.dead_loop_threshold :]
                    if all(h == recent[0] for h in recent):
                        self._finalize_budget()
                        return self._build_result(
                            False,
                            "|".join(obs_parts),
                            step,
                            f"检测到死循环（连续{self._config.dead_loop_threshold}次相同操作）",
                            trace,
                            last_tool_result,
                        )

                # ── 6. 对话压缩 ──
                messages = self._compress_messages(messages)

        # ── 达到最大步数 ──
        self._finalize_budget()
        return self._build_result(
            False,
            "已达到最大推理步数，请简化问题或补充信息",
            self.max_steps,
            "达到最大步数限制",
            trace,
            last_tool_result,
        )

    # ── Function Calling 支持探测 ─────────────────────

    def _check_fc_support(self, openai_tools: list[dict[str, Any]]) -> bool:
        """探测后端 API 是否支持 Function Calling"""
        if self._fc_supported is not None:
            return self._fc_supported
        if not self.generator or not openai_tools:
            self._fc_supported = False
            return False
        try:
            probe = self.generator.chat_with_tools(
                messages=[{"role": "user", "content": "ping"}],
                tools=openai_tools,
                tool_choice="none",
                max_tokens=10,
            )
            self._fc_supported = probe.content is not None or probe.finish_reason in ("stop",)
        except Exception:
            self._fc_supported = False
        return self._fc_supported

    # ── 提示词 ──────────────────────────────────────────

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """从注册工具构建 OpenAI 格式的 tools 列表"""
        tools = [tool.openai_tool_schema for tool in self.tools.values()]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": "从知识库检索相关信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            }
        )
        return tools

    def _build_system_prompt(self, memory_context: str = "") -> str:
        base = _FUNCTION_CALLING_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        if memory_context:
            base += f"\n\n{memory_context}"
        return base

    # ── 规划阶段 ─────────────────────────────────────────



# ════════════════════════════════════════════════════════════════
#  四、ReAct 循环（文本协议，降级用）
# ════════════════════════════════════════════════════════════════

_REACT_SYSTEM_PROMPT = """\
你是一个智能助手，通过 ReAct（Reasoning + Acting）方式工作。

## 可用工具
{tool_descriptions}

## 内置命令
除了上述工具，你还可以使用以下内置命令：
- **retrieve**: 从知识库检索相关信息
  参数: {{"query": "搜索关键词", "top_k": 5}}
  当你需要知识库中的文档来回答问题时使用。

## 工作流程
每轮你输出以下两种格式之一：

### 格式1：调用工具/命令
Thought: 你的推理过程...
Action: 工具名或命令名
Action Input: {{"参数名": "参数值"}}

### 格式2：给出最终答案
Thought: 你的推理过程...
Final Answer: 你的完整回答

## 规则
1. 一次只调用一个工具或命令
2. 调用后会收到 Observation（结果），基于 Observation 继续推理
3. 信息足够时输出 Final Answer
4. 最多 {max_steps} 步
5. 如果多次尝试同一操作得到相同结果，停止并给出当前信息
6. 如果无法完成任务，在 Final Answer 中诚实说明"""


class ReActLoop(AgentLoopBase):
    """自实现的 ReAct（Reasoning + Acting）循环

    核心思想：让 LLM 通过「思考→行动→观察→再思考」的迭代过程完成任务。

    继承自 AgentLoopBase 的公共 Harness 能力：
      - 预算控制（BudgetTracker）
      - 上下文压缩（ContextCompressor）
      - 工具执行（_execute_action / _execute_tool_calls / _execute_retrieve）
      - Policy 检查（_check_tool_policy）
      - 结果封装（_build_result）

    本类只保留 ReAct 文本协议特有逻辑：
      - 文本格式解析（_parse_react_output）
      - system prompt 构建（_build_system_prompt）
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        generator=None,
        rag_pipeline=None,
        harness_config: AgentHarnessConfig | None = None,
    ):
        super().__init__(
            tools=tools,
            harness_config=harness_config,
            generator=generator,
            rag_pipeline=rag_pipeline,
        )

    def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """执行 ReAct 循环"""
        context = context or {}

        self._init_budget()

        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
            {"role": "user", "content": self._build_user_message(query, context)},
        ]

        trace: list[dict[str, Any]] = []
        dead_loop_window: list[tuple[str, str, str]] = []
        last_tool_result = None

        for step in range(1, self.max_steps + 1):
            # ── 预算：墙钟 ──
            reason = self._check_wall_clock()
            if reason:
                self._finalize_budget()
                return self._build_result(False, reason, step, reason, trace, last_tool_result)

            # ── 1. LLM 推理 ──
            response = self.generator.chat(messages, temperature=0.3, max_tokens=2048)
            messages.append({"role": "assistant", "content": response})

            # ── 预算：步数 + Token ──
            step_tokens = BudgetTracker.estimate_tokens(response)
            reason = self._record_step(step_tokens)
            if reason:
                self._finalize_budget()
                return self._build_result(False, reason, step, reason, trace, last_tool_result)

            # ── 2. 解析 ──
            parsed = self._parse_react_output(response)
            if parsed is None:
                self._finalize_budget()
                return self._build_result(False, response, step, "输出格式解析失败", trace, last_tool_result)

            thought = parsed.get("thought", "")
            action = parsed.get("action")
            action_input = parsed.get("action_input")
            final_answer = parsed.get("final_answer")

            trace.append({"step": step, "thought": thought, "action": action, "action_input": action_input})

            # ── 3. Final Answer ──
            if final_answer is not None:
                self._finalize_budget()
                return self._build_result(True, final_answer, step, "正常完成", trace, last_tool_result)

            # ── 4. 执行 Action ──
            if action and action in self.tools:
                reason = self._record_tool_calls(1)
                if reason:
                    self._finalize_budget()
                    return self._build_result(False, reason, step, reason, trace, last_tool_result)

            observation = self._execute_action(action, action_input, query, context)
            observation = self._compress_observation(observation, action or "")

            if action in self.tools and isinstance(action_input, dict):
                try:
                    parsed_obs = json.loads(observation) if isinstance(observation, str) else observation
                    if isinstance(parsed_obs, dict):
                        last_tool_result = parsed_obs
                except (json.JSONDecodeError, TypeError):
                    pass

            messages.append({"role": "user", "content": f"Observation: {observation}"})

            # ── 5. 死循环检测 ──
            sig = (action or "", str(action_input or ""), str(observation)[:300])
            dead_loop_window.append(sig)
            if len(dead_loop_window) >= self._config.dead_loop_threshold:
                recent = dead_loop_window[-self._config.dead_loop_threshold :]
                if all(h == recent[0] for h in recent):
                    self._finalize_budget()
                    return self._build_result(
                        False,
                        observation,
                        step,
                        f"检测到死循环（连续{self._config.dead_loop_threshold}次相同操作）",
                        trace,
                        last_tool_result,
                    )

            # ── 6. 对话压缩 ──
            messages = self._compress_messages(messages)

        # ── 达到最大步数 ──
        self._finalize_budget()
        return self._build_result(
            False,
            "已达到最大推理步数，请简化问题或补充信息",
            self.max_steps,
            "达到最大步数限制",
            trace,
            last_tool_result,
        )

    # ── ReAct 文本协议特有 ───────────────────────────────

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """构建系统提示词（含工具的 JSON Schema 描述）"""
        lines = []
        for name, tool in self.tools.items():
            schema = tool.get_schema()
            desc = schema.get("description", "")
            params = schema.get("parameters", {}).get("properties", {})
            required = schema.get("parameters", {}).get("required", [])
            param_lines = []
            for pname, pinfo in params.items():
                rqd = "（必填）" if pname in required else "（可选）"
                param_lines.append(f"    {pname} {rqd}: {pinfo.get('description', '')}")
            lines.append(f"### {name}")
            lines.append(f"描述：{desc}")
            if param_lines:
                lines.append("参数：")
                lines.extend(param_lines)
            lines.append("")

        base_prompt = _REACT_SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(lines) or "（无）",
            max_steps=self.max_steps,
        )
        if memory_context:
            base_prompt += f"\n\n{memory_context}"
        return base_prompt

    @staticmethod
    def _parse_react_output(text: str) -> dict[str, Any] | None:
        """解析 ReAct 格式输出：Thought / Action / Action Input / Final Answer"""
        final_m = re.search(r"Final Answer:\s*(.*?)$", text, re.DOTALL)
        if final_m:
            thought = ""
            thought_m = re.search(r"Thought:\s*(.*?)(?=\nFinal Answer:)", text, re.DOTALL)
            if thought_m:
                thought = thought_m.group(1).strip()
            return {"thought": thought, "final_answer": final_m.group(1).strip(), "action": None, "action_input": None}

        action_m = re.search(r"Action:\s*(\S+)", text)
        if action_m:
            thought = ""
            thought_m = re.search(r"Thought:\s*(.*?)(?=\nAction:)", text, re.DOTALL)
            if thought_m:
                thought = thought_m.group(1).strip()
            action_input = None
            ai_m = re.search(r'Action Input:\s*(\{.*\}|".*?")', text, re.DOTALL)
            if ai_m:
                raw = ai_m.group(1).strip()
                try:
                    action_input = json.loads(raw)
                except json.JSONDecodeError:
                    action_input = raw.strip("\"'")
            return {
                "thought": thought,
                "action": action_m.group(1).strip(),
                "action_input": action_input,
                "final_answer": None,
            }

        return None


# ── 辅助函数 ─────────────────────────────────────────



# ════════════════════════════════════════════════════════════════
#  四、Agent 主控制器（ReAct 增强版）
# ════════════════════════════════════════════════════════════════


class Agent:
    """Agent 主控制器

    流程：
      用户请求 → LLMIntentClassifier（意图分类）
                │  ├─ pe_diagnosis → FunctionCallingLoop → 诊断工具
                │  ├─ report_generate → FunctionCallingLoop → 检索 + 报告生成
                │  └─ normal_query → 返回常规 RAG 流程
                │
                └─ IntentAnalyzer（规则兜底，LLM 不可用时自动切换）

    面试价值：
      - LLM 分类 vs 规则分类的优雅降级
      - Function Calling 循环作为工具执行的统一引擎（原生 OpenAI tools 支持）
      - 支持并行工具调用，API 不支持时自动降级到 ReAct 文本协议
      - 向后兼容的接口设计
    """

    def __init__(self, rag_pipeline=None, memory_manager=None, harness_config: AgentHarnessConfig | None = None):
        self.rag_pipeline = rag_pipeline
        self.tools: dict[str, Tool] = {}
        self.memory_manager: MemoryManager | None = memory_manager
        self._harness_config = harness_config or AgentHarnessConfig()
        self._register_default_tools()

        # LLM 驱动意图分类器（共享 pipeline 的 generator）
        self.intent_classifier = LLMIntentClassifier(
            generator=rag_pipeline.generator if rag_pipeline else None,
        )

        # Function Calling 循环（延迟初始化，内部含 ReActLoop 降级）
        self._react: FunctionCallingLoop | None = None

    @property
    def _react_loop(self) -> FunctionCallingLoop:
        if self._react is None:
            self._react = FunctionCallingLoop(
                tools=self.tools,
                generator=self.rag_pipeline.generator if self.rag_pipeline else None,
                rag_pipeline=self.rag_pipeline,
                harness_config=self._harness_config,
            )
        return self._react

    def _register_default_tools(self):
        """注册内置工具"""
        self.register_tool(ReportGenerator())
        self.register_tool(DiagnosisTool())

    def register_tool(self, tool: Tool) -> None:
        """注册一个新工具"""
        self.tools[tool.name] = tool
        if hasattr(tool, "set_generator") and self.rag_pipeline:
            tool.set_generator(self.rag_pipeline.generator)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.get_schema() for tool in self.tools.values()]

    # ── 主入口 ─────────────────────────────────────────

    def process(
        self,
        query: str,
        top_k: int = 8,
        file_path: str | None = None,
        use_llm_classifier: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """处理用户请求

        Args:
            query:             用户问题
            top_k:             检索数量
            file_path:         影像文件路径（仅诊断时需要）
            use_llm_classifier: 使用 LLM 分类（False = 纯规则）
            session_id:        会话 ID（启用记忆系统时传入）

        Returns:
            {
                "agent_handled": bool,
                "intent": dict,       # 意图分析结果
                "tool": str|None,
                "result": dict,       # 工具执行结果
                "report_type": str|None,
                "react_trace": [...]|None,
                "memory_used": bool,  # 是否使用了记忆
            }
        """
        start = time.monotonic()

        # ── 追踪 span ──
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.process") as span:
            span.set_attribute("query_length", len(query))
            span.set_attribute("use_llm_classifier", use_llm_classifier)
            span.set_attribute("has_session_id", session_id is not None)

            # ── 0. 构建记忆上下文 ──
            memory_context = ""
            effective_sid = session_id or "default"
            mm = self.memory_manager

            if mm is not None and session_id:
                try:
                    memory_context = mm.build_context(effective_sid, query)
                    span.set_attribute("memory_context_length", len(memory_context))
                except Exception:
                    memory_context = ""

            # ── 1. 意图分类（LLM 优先，规则兜底） ──
            if use_llm_classifier:
                intent = self.intent_classifier.classify(query)
            else:
                intent = IntentAnalyzer.analyze(query)
                intent["llm_classified"] = False

            intent_name = (
                intent.get("intent", intent.get("intent_str", "unknown")) if isinstance(intent, dict) else str(intent)
            )
            span.set_attribute("intent", intent_name if isinstance(intent_name, str) else str(intent_name))
            span.set_attribute("llm_classified", intent.get("llm_classified", False))
            self._print_header(intent)

            # ── 2. 常规问答 → 不处理，交还给 RAG 流程 ──
            if intent["intent"] == "normal_query":
                if mm is not None and session_id:
                    try:
                        mm.remember(session_id=effective_sid, query=query, answer=query, intent_info=intent)
                    except Exception:
                        pass
                elapsed = time.monotonic() - start
                span.set_attribute("duration_ms", round(elapsed * 1000, 1))
                return {
                    "agent_handled": False,
                    "intent": intent,
                    "message": "未匹配到工具，交由常规 RAG 流程处理",
                }

            # ── 3. 需要工具 ──
            context = {"file_path": file_path, "top_k": top_k}
            if intent.get("report_type"):
                context["report_type"] = intent["report_type"]

            # 检查是否有 generator，无则走直接工具调用（规则兜底）
            generator = self._react_loop.generator
            if generator is None:
                result = self._handle_without_react(intent, query, top_k, file_path)
            else:
                with tracer.start_as_current_span("agent.react_loop") as react_span:
                    react_result = self._react_loop.run(
                        query, context=context, memory_context=memory_context, session_id=effective_sid
                    )
                    steps = len(react_result.get("trace", []))
                    react_span.set_attribute("steps", steps)
                    react_span.set_attribute("termination", react_result.get("termination", "unknown"))
                    record_agent_steps(steps)
                result = self._wrap_react_result(intent, react_result)

            # ── 5. 记录本次交互到记忆系统 ──
            answer_text = ""
            if isinstance(result.get("result"), dict):
                answer_text = str(result["result"].get("report", result["result"].get("formatted_report", "")))
            elif isinstance(result.get("result"), str):
                answer_text = result["result"]

            if mm is not None and session_id:
                try:
                    mm.remember(session_id=effective_sid, query=query, answer=answer_text or query, intent_info=intent)
                except Exception:
                    pass
                result["memory_used"] = True

            elapsed = time.monotonic() - start
            span.set_attribute("duration_ms", round(elapsed * 1000, 1))
            span.set_attribute("agent_handled", result.get("agent_handled", False))
            record_request("agent_process", 200, elapsed)

            return result

    def _handle_without_react(
        self,
        intent: dict[str, Any],
        query: str,
        top_k: int = 8,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """无 LLM 时的直接工具调用（ReAct 不可用时的兜底）"""
        tool_name = intent.get("tool_name")
        if not tool_name or tool_name not in self.tools:
            return {
                "agent_handled": False,
                "intent": intent,
                "message": f"工具 '{tool_name}' 未注册",
            }

        tool = self.tools[tool_name]

        if intent["intent"] == "pe_diagnosis":
            if not file_path:
                return {
                    "agent_handled": True,
                    "intent": intent,
                    "tool": tool_name,
                    "result": {"success": False, "error": "诊断需要 CTPA 影像文件路径"},
                    "needs_file": True,
                }
            result = tool.run(file_path=file_path)
            return {
                "agent_handled": True,
                "intent": intent,
                "tool": tool_name,
                "result": result,
                "react_trace": None,
                "react_steps": 0,
                "react_termination": "无 LLM，直接调用工具",
            }

        if intent["intent"] == "report_generate":
            # 检索 + 报告生成（旧版流程）
            report_type = intent.get("report_type", "deployment")
            retrieved = []
            if self.rag_pipeline and self.rag_pipeline.retriever:
                retrieved = self.rag_pipeline.retriever.retrieve(query, top_k=top_k)

            content_parts = []
            for i, chunk in enumerate(retrieved, 1):
                meta = chunk["metadata"]
                content_parts.append(f"### 来源 [{i}]: {meta.get('filename', '未知')}\n{chunk['text']}\n")
            content = "\n\n".join(content_parts) if content_parts else query

            result = tool.run(
                report_type=report_type,
                content=content,
                topic=query,
                context={
                    "sources_summary": "知识库检索" if retrieved else "用户提供",
                    "source_count": len(retrieved),
                },
            )
            return {
                "agent_handled": True,
                "intent": intent,
                "tool": tool_name,
                "report_type": report_type,
                "result": result,
                "react_trace": None,
                "react_steps": 0,
                "react_termination": "无 LLM，直接调用工具",
            }

        return {"agent_handled": False, "intent": intent, "message": "未知意图"}

    # ── 内部方法 ───────────────────────────────────────

    def _print_header(self, intent: dict[str, Any]) -> None:
        """打印调度信息"""
        print("\n" + "=" * 60)
        print("  🧠 Agent 调度引擎 (ReAct)")
        print("=" * 60)
        print(f"  📋 意图: {intent['intent']}")
        method = "LLM" if intent.get("llm_classified") else "规则(兜底)"
        print(f"  🤖 分类方式: {method}")
        if intent.get("report_type"):
            print(f"  📋 子类型: {intent['report_type']}")
        if intent.get("tool_name"):
            print(f"  🔧 匹配工具: {intent['tool_name']}")

    def _wrap_react_result(
        self,
        intent: dict[str, Any],
        react_result: dict[str, Any],
    ) -> dict[str, Any]:
        """将 ReAct 结果封装为与旧版兼容的返回格式"""
        base = {
            "agent_handled": True,
            "intent": intent,
            "tool": intent.get("tool_name"),
            "report_type": intent.get("report_type"),
            "react_trace": react_result.get("trace"),
            "react_steps": react_result.get("steps"),
            "react_termination": react_result.get("termination_reason"),
        }

        tool_result = react_result.get("tool_result")

        # 诊断结果：优先使用工具的结构化输出
        if intent["intent"] == "pe_diagnosis":
            if isinstance(tool_result, dict) and tool_result.get("success") is not None:
                base["result"] = tool_result
            else:
                base["result"] = {
                    "success": react_result.get("success", False),
                    "formatted_report": str(react_result.get("result", "")),
                }
            return base

        # 报告结果：优先使用工具的结构化输出
        if intent["intent"] == "report_generate":
            if isinstance(tool_result, dict) and "report" in tool_result:
                base["result"] = tool_result
            else:
                base["result"] = {
                    "success": react_result.get("success", False),
                    "report": str(react_result.get("result", "")),
                }
            return base

        return base

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
from typing import Any

from .memory import MemoryManager
from .monitoring.metrics import record_agent_steps, record_request

# ── 可观测性 ────────────────────────────────────────
from .monitoring.tracing import get_tracer
from .tools.base import Tool
from .tools.diagnosis_tool import DiagnosisTool
from .tools.report_generator import ReportGenerator

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
            "部署报告",
            "部署文档",
            "部署方案",
            "部署流程",
            "上线报告",
            "发布报告",
            "部署说明",
            "部署步骤",
            "部署计划",
            "怎么部署",
            "如何部署",
            "部署到",
            "部署在",
            "安装部署",
            "环境搭建",
            "搭建环境",
            "部署文档",
            "部署方案",
            "写一份部署",
            "deployment",
            "deploy report",
            "release report",
        ],
        "troubleshoot": [
            "排查报告",
            "问题排查",
            "故障报告",
            "故障排查",
            "排查文档",
            "怎么解决",
            "如何解决",
            "是什么原因",
            "为什么报错",
            "故障分析",
            "根因分析",
            "问题分析",
            "排查步骤",
            "写一份排查",
            "排查方案",
            "troubleshoot",
            "troubleshooting",
            "incident report",
            "错误",
            "报错",
            "失败",
            "异常",
            "崩溃",
            "宕机",
        ],
        "meeting": [
            "会议纪要",
            "会议记录",
            "会议总结",
            "会议摘要",
            "会议备忘",
            "会议讨论",
            "写纪要",
            "整理成纪要",
            "生成纪要",
            "meeting minutes",
            "meeting notes",
            "纪要",
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
        "栓塞诊断",
        "pe诊断",
        "pe检测",
        "pe预测",
        "诊断影像",
        "诊断ct",
        "诊断ctpa",
        "影像诊断",
        "分析影像",
        "分析扫描",
        "分析ct",
        "分析ctpa",
        "读片",
        "读ct",
        "阅片",
        "预测风险",
        "风险评估",
        "风险预测",
        "是不是肺栓塞",
        "是否肺栓塞",
        "有没有肺栓塞",
        "诊断一下",
        "检测一下",
        "预测一下",
        "diagnose",
        "pulmonary embolism diagnosis",
        "pe diagnosis",
        "embolism detection",
        "pe detection",
        "ctpa analysis",
        "ct analysis",
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
#  三、Function Calling 循环（原生 OpenAI tools 支持）
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


class FunctionCallingLoop:
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
    """

    MAX_STEPS = 10
    DEAD_LOOP_THRESHOLD = 3

    def __init__(
        self,
        tools: dict[str, "Tool"],
        generator=None,
        rag_pipeline=None,
        max_steps: int = MAX_STEPS,
    ):
        self.tools = tools
        self.generator = generator
        self.rag_pipeline = rag_pipeline
        self.max_steps = max_steps
        self._fc_supported: bool | None = None  # 缓存探测结果

    # ── 公共入口 ──────────────────────────────────────

    def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        """执行 Function Calling 循环

        返回格式与 ReActLoop.run() 完全一致，保证 Agent 主流程无需改动。

        Returns:
            success, result, tool_result, steps, termination_reason, trace
        """
        context = context or {}
        openai_tools = self._build_openai_tools()

        # ── 探测 Function Calling 支持 ──
        if not self._check_fc_support(openai_tools):
            # 不支持 → 内部降级到 ReActLoop
            fallback = ReActLoop(
                tools=self.tools,
                generator=self.generator,
                rag_pipeline=self.rag_pipeline,
                max_steps=self.max_steps,
            )
            return fallback.run(query, context=context, memory_context=memory_context)

        # ── 初始化消息列表 ──
        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
            {"role": "user", "content": self._build_user_message(query, context)},
        ]

        trace: list[dict[str, Any]] = []
        dead_loop_window: list[tuple[str, str]] = []
        last_tool_result: dict[str, Any] | None = None

        for step in range(1, self.max_steps + 1):
            # ── 1. 调用 LLM（带 tools 参数） ──
            response = self.generator.chat_with_tools(
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2048,
                parallel_tool_calls=True,
            )

            # ── 2. 追加 assistant 消息（含 tool_calls，如果有） ──
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

            # ── 3. LLM 选择直接回答（无 tool_calls）→ 最终答案 ──
            if not response.tool_calls and response.content:
                return self._build_result(
                    True,
                    response.content,
                    step,
                    "正常完成",
                    trace,
                    last_tool_result,
                )

            # ── 4. 执行所有 tool_calls（并行） ──
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls, query, context)

                # 将每个结果以 tool role 追加
                for tc, tr in zip(response.tool_calls, tool_results):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tr["observation"],
                        }
                    )

                # 保存最新的结构化结果
                for tr in tool_results:
                    if tr.get("structured"):
                        last_tool_result = tr["structured"]

                # ── 5. 死循环检测 ──
                sig_parts = [tc["function"]["name"] for tc in response.tool_calls]
                obs_parts = [tr["observation"][:300] for tr in tool_results]
                dead_loop_window.append(("|".join(sig_parts), "|".join(obs_parts)))
                if len(dead_loop_window) >= self.DEAD_LOOP_THRESHOLD:
                    recent = dead_loop_window[-self.DEAD_LOOP_THRESHOLD :]
                    if all(h == recent[0] for h in recent):
                        return self._build_result(
                            False,
                            "|".join(obs_parts),
                            step,
                            f"检测到死循环（连续{self.DEAD_LOOP_THRESHOLD}次相同操作）",
                            trace,
                            last_tool_result,
                        )

        # ── 达到最大步数 ──
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
        """探测后端 API 是否支持 Function Calling

        发一条最小 probe 请求，tool_choice="none" 强制模型不调工具。
        成功 → 支持；失败 → 降级到 ReActLoop。
        结果缓存在 self._fc_supported，只探测一次。
        """
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

    # ── 工具执行（并行） ──────────────────────────────

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        query: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """并行执行多个 tool_calls

        用 ThreadPoolExecutor 执行 I/O-bound 的工具调用。
        结果按 tool_calls 的顺序返回。

        Returns:
            [{"observation": str, "structured": dict|None}, ...]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def execute_one(tc: dict[str, Any]) -> dict[str, Any]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}

            # 注入外部上下文参数
            for key in ("file_path",):
                if key not in args and context.get(key):
                    args[key] = context[key]

            # 内置命令：知识库检索
            if name == "retrieve":
                obs = self._execute_retrieve(args, query)
                return {"observation": obs, "structured": None}

            # 注册工具
            if name in self.tools:
                tool = self.tools[name]
                try:
                    result = tool.run(**args)
                    obs = json.dumps(result, ensure_ascii=False, default=str)[:3000]
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

    # ── 辅助方法 ──────────────────────────────────────

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """从注册工具构建 OpenAI 格式的 tools 列表

        包含所有注册工具 + 内置 retrieve 命令。
        """
        tools = [tool.openai_tool_schema for tool in self.tools.values()]

        # 内置：知识库检索
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": "从知识库检索相关信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回结果数量",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )

        return tools

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """构建系统提示词

        与 ReActLoop 不同，不把工具 Schema 内联进 prompt 文本，
        而是由 API 的 tools 参数提供。prompt 只保留行为规则。
        """
        base = _FUNCTION_CALLING_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        if memory_context:
            base += f"\n\n{memory_context}"
        return base

    @staticmethod
    def _build_user_message(query: str, context: dict[str, Any]) -> str:
        extra = ""
        if context.get("file_path"):
            extra += f"\n用户已提供文件路径：{context['file_path']}"
        if context.get("report_type"):
            extra += f"\n用户指定报告类型：{context['report_type']}"
        return f"用户问题：{query}{extra}"

    def _execute_retrieve(self, args: dict[str, Any], default_query: str) -> str:
        """执行知识库检索（与 ReActLoop 逻辑一致）"""
        if not self.rag_pipeline or not self.rag_pipeline.retriever:
            return "检索功能不可用（未连接知识库）"

        q = args.get("query", default_query)
        k = args.get("top_k", 5)

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

    @staticmethod
    def _build_result(
        success: bool,
        result: Any,
        steps: int,
        reason: str,
        trace: list[dict[str, Any]],
        tool_result: Any = None,
    ) -> dict[str, Any]:
        """封装结果（与 ReActLoop 格式一致）"""
        return {
            "success": success,
            "result": result,
            "tool_result": tool_result,
            "steps": steps,
            "termination_reason": reason,
            "trace": trace,
        }


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


class ReActLoop:
    """自实现的 ReAct（Reasoning + Acting）循环

    核心思想：让 LLM 通过「思考→行动→观察→再思考」的迭代过程完成任务。

      1. Thought:  LLM 分析当前状态，决定下一步做什么
      2. Action:   调用工具或内置命令
      3. Observation: 接收执行结果
      4. → 回到 1，直到 LLM 输出 Final Answer 或触发终止条件

    面试价值：
      - 展示对 Agent 核心机制的自实现能力（不依赖 LangChain）
      - 理解 ReAct 如何解决 LLM 单次推理的局限性
      - 包含完整的工业级终止条件设计

    终止条件：
      1. LLM 输出 Final Answer → 正常结束
      2. 达到 MAX_STEPS 上限   → 超时终止
      3. 死循环检测            → 连续 N 次相同操作且结果相同
    """

    MAX_STEPS = 10
    DEAD_LOOP_THRESHOLD = 3

    def __init__(
        self,
        tools: dict[str, Tool],
        generator=None,
        rag_pipeline=None,
        max_steps: int = MAX_STEPS,
    ):
        self.tools = tools
        self.generator = generator
        self.rag_pipeline = rag_pipeline
        self.max_steps = max_steps

    def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        """执行 ReAct 循环

        Args:
            query:          用户问题
            context:        额外上下文（如 file_path、top_k 等）
            memory_context: 记忆上下文文本（由 MemoryManager 构建，注入到 system prompt）

        Returns:
            {
                "success": bool,
                "result": Any,            # 最终回答或工具输出
                "tool_result": Any|None,  # 最近一次工具执行的结构化结果
                "steps": int,             # 实际执行步数
                "termination_reason": str,# 终止原因
                "trace": [...],           # 完整的思考-行动轨迹
            }
        """
        context = context or {}

        # ── 初始化消息列表 ──
        messages = [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
            {"role": "user", "content": self._build_user_message(query, context)},
        ]

        trace: list[dict[str, Any]] = []
        dead_loop_window: list[tuple[str, str, str]] = []
        last_tool_result = None  # 保存最近一次工具的结构化输出

        for step in range(1, self.max_steps + 1):
            # ── 1. LLM 推理 ──
            response = self.generator.chat(messages, temperature=0.3, max_tokens=2048)
            messages.append({"role": "assistant", "content": response})

            # ── 2. 解析输出 ──
            parsed = self._parse_react_output(response)
            if parsed is None:
                return self._build_result(
                    False,
                    response,
                    step,
                    "输出格式解析失败",
                    trace,
                    last_tool_result,
                )

            thought = parsed.get("thought", "")
            action = parsed.get("action")
            action_input = parsed.get("action_input")
            final_answer = parsed.get("final_answer")

            trace.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                }
            )

            # ── 3. Final Answer → 正常结束 ──
            if final_answer is not None:
                return self._build_result(
                    True,
                    final_answer,
                    step,
                    "正常完成",
                    trace,
                    last_tool_result,
                )

            # ── 4. 执行 Action → 得到 Observation ──
            observation = self._execute_action(action, action_input, query, context)
            # 如果执行的是注册工具，保存结构化结果
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
            if len(dead_loop_window) >= self.DEAD_LOOP_THRESHOLD:
                recent = dead_loop_window[-self.DEAD_LOOP_THRESHOLD :]
                if all(h == recent[0] for h in recent):
                    return self._build_result(
                        False,
                        observation,
                        step,
                        f"检测到死循环（连续{self.DEAD_LOOP_THRESHOLD}次相同操作）",
                        trace,
                        last_tool_result,
                    )

        # ── 达到最大步数 ──
        return self._build_result(
            False,
            "已达到最大推理步数，请简化问题或补充信息",
            self.max_steps,
            "达到最大步数限制",
            trace,
            last_tool_result,
        )

    # ── 提示词构建 ──────────────────────────────────────

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """构建系统提示词（含所有工具的 JSON Schema 描述 + 可选记忆上下文）"""
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
    def _build_user_message(query: str, context: dict[str, Any]) -> str:
        extra = ""
        if context.get("file_path"):
            extra += f"\n用户已提供文件路径：{context['file_path']}"
        if context.get("report_type"):
            extra += f"\n用户指定报告类型：{context['report_type']}"
        return f"用户问题：{query}{extra}"

    # ── 输出解析 ───────────────────────────────────────

    @staticmethod
    def _parse_react_output(text: str) -> dict[str, Any] | None:
        """解析 ReAct 格式输出：Thought / Action / Action Input / Final Answer"""
        # 优先检查 Final Answer
        final_m = re.search(
            r"Final Answer:\s*(.*?)$",
            text,
            re.DOTALL,
        )
        if final_m:
            thought = ""
            thought_m = re.search(r"Thought:\s*(.*?)(?=\nFinal Answer:)", text, re.DOTALL)
            if thought_m:
                thought = thought_m.group(1).strip()
            return {
                "thought": thought,
                "final_answer": final_m.group(1).strip(),
                "action": None,
                "action_input": None,
            }

        # 检查 Action
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
                    # 可能是裸字符串
                    action_input = raw.strip("\"'")

            return {
                "thought": thought,
                "action": action_m.group(1).strip(),
                "action_input": action_input,
                "final_answer": None,
            }

        return None

    # ── Action 执行 ────────────────────────────────────

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
            # 注入外部上下文参数
            for key in ("file_path",):
                if key not in params and context.get(key):
                    params[key] = context[key]
            try:
                result = tool.run(**params)
                return json.dumps(result, ensure_ascii=False, default=str)[:3000]
            except Exception as e:
                return f"工具调用失败：{e}"

        return f"未知命令：{action}。可用工具：{list(self.tools.keys())}，内置命令：['retrieve']"

    def _execute_retrieve(self, action_input: Any, default_query: str) -> str:
        """执行知识库检索"""
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

    # ── 结果封装 ───────────────────────────────────────

    @staticmethod
    def _build_result(
        success: bool,
        result: Any,
        steps: int,
        reason: str,
        trace: list[dict[str, Any]],
        tool_result: Any = None,
    ) -> dict[str, Any]:
        return {
            "success": success,
            "result": result,
            "tool_result": tool_result,
            "steps": steps,
            "termination_reason": reason,
            "trace": trace,
        }


# ── 辅助函数 ─────────────────────────────────────────


def _remember_if_needed(
    mm: "MemoryManager",
    effective_sid: str,
    session_id: str | None,
    query: str,
    intent: dict[str, Any],
    answer_text: str,
) -> None:
    """条件性调用记忆记录，避免 repeated try/except"""
    if mm is None or not session_id:
        return
    try:
        mm.remember(
            session_id=effective_sid,
            query=query,
            answer=answer_text or query,
            intent_info=intent,
        )
    except Exception:
        pass


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

    def __init__(self, rag_pipeline=None, memory_manager=None):
        self.rag_pipeline = rag_pipeline
        self.tools: dict[str, Tool] = {}
        self.memory_manager: MemoryManager | None = memory_manager
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
                _remember_if_needed(mm, effective_sid, session_id, query, intent, "")
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
            generator = self._react_loop.generator if self._react else None
            if generator is None:
                result = self._handle_without_react(intent, query, top_k, file_path)
            else:
                with tracer.start_as_current_span("agent.react_loop") as react_span:
                    react_result = self._react_loop.run(query, context=context, memory_context=memory_context)
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

            _remember_if_needed(mm, effective_sid, session_id, query, intent, answer_text)
            if mm is not None and session_id:
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

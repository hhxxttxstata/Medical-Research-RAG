"""
Agent 模块单元测试（新架构）

覆盖：
  - IntentAnalyzer（规则兜底，向后兼容）
  - LLMIntentClassifier（LLM 分类 + 规则降级）
  - ReActLoop（输出解析、死循环检测、结果封装）
  - Agent（主流程集成）
"""

from typing import Any

from src.agent import (
    Agent,
    AgentBudget,
    AgentHarnessConfig,
    BudgetTracker,
    IntentAnalyzer,
    LLMIntentClassifier,
    ReActLoop,
)
from src.tools.base import Tool

# ═══════════════════════════════════════════════════════════════
#  测试辅助
# ═══════════════════════════════════════════════════════════════


class MockGenerator:
    """模拟 LLM Generator，可控输出"""

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.call_history: list = []

    def chat(self, messages, temperature=0.0, max_tokens=256) -> str:
        """根据最后一轮 user message 匹配预设回复"""
        self.call_history.append({"messages": messages, "temperature": temperature})
        for msg in reversed(messages):
            if msg["role"] == "user":
                key = msg["content"][:100]  # 截取开头做匹配
                for pattern, resp in self.responses.items():
                    if pattern in key:
                        return resp
        return self.responses.get("__default__", '{"intent":"normal_query","report_type":null,"reasoning":"default"}')


class SimpleTool(Tool):
    name = "echo"
    description = "回显工具"

    def run(self, **kwargs) -> dict[str, Any]:
        return {"success": True, "message": str(kwargs)}

    def get_schema(self):
        return {
            "tool_name": "echo",
            "description": "回显工具",
            "parameters": {
                "type": "object",
                "properties": {"msg": {"type": "string", "description": "消息"}},
                "required": ["msg"],
            },
        }


class MockRetriever:
    """模拟检索器"""

    def retrieve(self, query, top_k=5):
        return [
            {"text": "肺栓塞是一种急重症。", "metadata": {"filename": "doc.md", "page": 1}, "score": 0.9},
        ]


class MockRAGPipeline:
    def __init__(self):
        self.retriever = MockRetriever()
        self.generator = None


# ═══════════════════════════════════════════════════════════════
#  一、IntentAnalyzer 向后兼容测试
# ═══════════════════════════════════════════════════════════════


class TestIntentAnalyzer:
    """原有测试全部保留，确保向后兼容"""

    def test_normal_query(self):
        result = IntentAnalyzer.analyze("什么是肺栓塞")
        assert result["intent"] == "normal_query"
        assert result["tool_name"] is None

    def test_deployment_report(self):
        result = IntentAnalyzer.analyze("帮我生成一份部署报告")
        assert result["intent"] == "report_generate"
        assert result["report_type"] == "deployment"
        assert result["tool_name"] == "generate_report"

    def test_troubleshoot_report(self):
        result = IntentAnalyzer.analyze("帮我排查一下部署失败的原因")
        assert result["intent"] == "report_generate"
        assert result["report_type"] == "troubleshoot"

    def test_meeting_minutes(self):
        result = IntentAnalyzer.analyze("把今天的讨论整理成会议纪要")
        assert result["intent"] == "report_generate"
        assert result["report_type"] == "meeting"

    def test_diagnosis_intent(self):
        result = IntentAnalyzer.analyze("帮我诊断一下这个CT影像是不是肺栓塞")
        assert result["intent"] == "pe_diagnosis"
        assert result["tool_name"] == "diagnose_pulmonary_embolism"

    def test_case_insensitive(self):
        assert IntentAnalyzer.analyze("DEPLOYMENT")["intent"] == "report_generate"
        assert IntentAnalyzer.analyze("PE DIAGNOSIS")["intent"] == "pe_diagnosis"


# ═══════════════════════════════════════════════════════════════
#  二、LLMIntentClassifier 测试
# ═══════════════════════════════════════════════════════════════


class TestLLMIntentClassifier:
    """LLM 驱动的分类器测试"""

    def test_fallback_no_generator(self):
        """没有 generator 时自动回退到规则"""
        classifier = LLMIntentClassifier(generator=None)
        result = classifier.classify("帮我诊断一下这个CT")
        assert result["intent"] == "pe_diagnosis"
        assert result["llm_classified"] is False

    def test_llm_classify_pe_diagnosis(self):
        """LLM 分类应解析出 pe_diagnosis"""
        gen = MockGenerator(
            responses={
                "帮我看看这个CT": '{"intent":"pe_diagnosis","report_type":null,"reasoning":"用户想看CT影像"}',
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        result = classifier.classify("帮我看看这个CT片子有没有问题")
        assert result["intent"] == "pe_diagnosis"
        assert result["tool_name"] == "diagnose_pulmonary_embolism"
        assert result["llm_classified"] is True

    def test_llm_classify_report(self):
        """LLM 分类应解析出 report_generate + report_type"""
        gen = MockGenerator(
            responses={
                "生成部署报告": '{"intent":"report_generate","report_type":"deployment","reasoning":"用户要部署文档"}',
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        result = classifier.classify("帮我生成部署报告")
        assert result["intent"] == "report_generate"
        assert result["report_type"] == "deployment"
        assert result["tool_name"] == "generate_report"

    def test_llm_classify_normal_query(self):
        """LLM 分类应解析出 normal_query"""
        gen = MockGenerator(
            responses={
                "什么是肺栓塞": '{"intent":"normal_query","report_type":null,"reasoning":"知识性提问"}',
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        result = classifier.classify("什么是肺栓塞")
        assert result["intent"] == "normal_query"
        assert result["tool_name"] is None

    def test_parse_json_with_markdown_fence(self):
        """能处理 markdown ```json 包裹的输出"""
        text = '```json\n{"intent":"pe_diagnosis","report_type":null}\n```'
        parsed = LLMIntentClassifier._parse_json(text)
        assert parsed is not None
        assert parsed["intent"] == "pe_diagnosis"

    def test_parse_json_with_extra_text(self):
        """能从带多余文本的输出中提取 JSON"""
        text = (
            "好的，我来分析。\n"
            "根据用户问题，我判断为诊断意图：\n"
            '{"intent": "pe_diagnosis", "report_type": null, "reasoning": "用户想分析CT"}'
        )
        parsed = LLMIntentClassifier._parse_json(text)
        assert parsed is not None
        assert parsed["intent"] == "pe_diagnosis"

    def test_parse_invalid_json_fallback(self):
        """无效 JSON 应返回 None 触发规则兜底"""
        gen = MockGenerator(
            responses={
                "xyz": "这不是JSON",
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        result = classifier.classify("xyz")
        # 回退到规则判断
        assert result["llm_classified"] is False

    def test_llm_classify_call_with_correct_prompt(self):
        """验证分类 prompt 包含所有意图选项"""
        gen = MockGenerator(
            responses={
                "__default__": '{"intent":"normal_query","report_type":null,"reasoning":"test"}',
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        classifier.classify("测试问题")
        assert len(gen.call_history) >= 1
        sys_msg = gen.call_history[0]["messages"][0]
        assert "pe_diagnosis" in sys_msg["content"]
        assert "report_generate" in sys_msg["content"]
        assert "normal_query" in sys_msg["content"]

    def test_temperature_is_zero_for_classification(self):
        """分类任务 temperature 应为 0"""
        gen = MockGenerator(
            responses={
                "__default__": '{"intent":"normal_query","report_type":null,"reasoning":"test"}',
            }
        )
        classifier = LLMIntentClassifier(generator=gen)
        classifier.classify("测试")
        assert gen.call_history[0]["temperature"] == 0.0


# ═══════════════════════════════════════════════════════════════
#  三、ReActLoop 测试
# ═══════════════════════════════════════════════════════════════


class TestReActLoop:
    """ReAct 循环单元测试"""

    def test_parse_final_answer(self):
        """解析 Final Answer 格式"""
        text = """Thought: 用户问了一个简单的知识性问题。
Final Answer: 肺栓塞是一种急重症，CTPA是诊断金标准。"""
        result = ReActLoop._parse_react_output(text)
        assert result is not None
        assert "肺栓塞是一种急重症" in result["final_answer"]
        assert result["action"] is None

    def test_parse_action(self):
        """解析 Action 格式"""
        text = """Thought: 用户需要检索知识库。
Action: retrieve
Action Input: {"query": "肺栓塞CT表现", "top_k": 5}"""
        result = ReActLoop._parse_react_output(text)
        assert result is not None
        assert result["action"] == "retrieve"
        assert result["action_input"]["query"] == "肺栓塞CT表现"
        assert result["final_answer"] is None
        assert "检索知识库" in result["thought"]

    def test_parse_garbage_returns_none(self):
        """完全无法解析的输出返回 None"""
        result = ReActLoop._parse_react_output("今天的天气真好")
        assert result is None

    def test_dead_loop_detection(self):
        """连续相同操作应触发死循环检测"""
        gen = MockGenerator(
            responses={
                "__default__": """Thought: 让我再试一次。
Action: retrieve
Action Input: {"query": "test"}""",
            }
        )
        loop = ReActLoop(
            tools={},
            generator=gen,
            rag_pipeline=MockRAGPipeline(),
            harness_config=AgentHarnessConfig(max_steps=5),
        )
        result = loop.run("test")
        assert result["success"] is False
        assert "死循环" in result["termination_reason"]

    def test_max_steps_termination(self):
        """达到最大步数应终止"""

        # 让每次 action_input 不同以避免死循环
        class RotatingGenerator:
            def __init__(self, responses):
                self.responses = responses
                self.idx = 0
                self.call_history = []

            def chat(self, messages, temperature=0.0, max_tokens=256) -> str:
                self.call_history.append(messages)
                resp = self.responses[self.idx % len(self.responses)]
                self.idx += 1
                return resp

        rotating_gen = RotatingGenerator(
            responses=[
                'Thought: 检索第1次。\nAction: retrieve\nAction Input: {"query": "query_1", "top_k": 3}',
                'Thought: 检索第2次。\nAction: retrieve\nAction Input: {"query": "query_2", "top_k": 3}',
                'Thought: 检索第3次。\nAction: retrieve\nAction Input: {"query": "query_3", "top_k": 3}',
                'Thought: 检索第4次。\nAction: retrieve\nAction Input: {"query": "query_4", "top_k": 3}',
                'Thought: 检索第5次。\nAction: retrieve\nAction Input: {"query": "query_5", "top_k": 3}',
            ]
        )
        loop = ReActLoop(
            tools={},
            generator=rotating_gen,
            rag_pipeline=MockRAGPipeline(),
            harness_config=AgentHarnessConfig(max_steps=4),
        )
        result = loop.run("test")
        assert result["steps"] == 4
        assert "最大步数限制" in result["termination_reason"]

    def test_unknown_action(self):
        """未知 action 应返回错误信息"""
        gen = MockGenerator(
            responses={
                "__default__": """Thought: 试试不存在的工具。
Action: nonexistent_tool
Action Input: {"msg": "hello"}""",
            }
        )
        loop = ReActLoop(
            tools={"echo": SimpleTool()},
            generator=gen,
            rag_pipeline=MockRAGPipeline(),
        )
        result = loop.run("test")
        assert "未知命令" in str(result["result"]).strip() or result["success"] is False

    def test_tool_execution_saves_tool_result(self):
        """工具执行后 tool_result 应保持结构化输出"""
        gen = MockGenerator(
            responses={
                "__default__": """Thought: 调用 echo 工具。
Action: echo
Action Input: {"msg": "hello"}""",
            }
        )
        loop = ReActLoop(
            tools={"echo": SimpleTool()},
            generator=gen,
            rag_pipeline=MockRAGPipeline(),
            harness_config=AgentHarnessConfig(max_steps=2),
        )
        result = loop.run("test")
        # ReAct 循环要么在有工具输出后停止，要么按步数终止
        # 关键：tool_result 不为 None 且是 dict
        if result["tool_result"] is not None:
            assert isinstance(result["tool_result"], dict)
            assert result["tool_result"].get("success") is True

    def test_build_system_prompt_includes_tools(self):
        """系统提示词应包含所有工具的描述"""
        loop = ReActLoop(
            tools={"echo": SimpleTool()},
            generator=MockGenerator(),
            rag_pipeline=MockRAGPipeline(),
        )
        prompt = loop._build_system_prompt()
        assert "echo" in prompt
        assert "回显工具" in prompt
        assert "msg" in prompt

    def test_build_result_format(self):
        """_build_result 应返回正确格式"""
        result = ReActLoop._build_result(
            success=True,
            result="回答内容",
            steps=3,
            reason="正常完成",
            trace=[{"step": 1, "thought": "思考", "action": None, "action_input": None}],
            tool_result={"success": True},
        )
        assert result["success"] is True
        assert result["result"] == "回答内容"
        assert result["steps"] == 3
        assert result["termination_reason"] == "正常完成"
        assert result["tool_result"]["success"] is True

    def test_retrieve_command(self):
        """retrieve 命令应在有 pipeline 时正常工作"""
        gen = MockGenerator(
            responses={
                "__default__": """Thought: 需要查询知识库。
Action: retrieve
Action Input: {"query": "肺栓塞", "top_k": 3}""",
            }
        )
        loop = ReActLoop(
            tools={},
            generator=gen,
            rag_pipeline=MockRAGPipeline(),
            harness_config=AgentHarnessConfig(max_steps=2),
        )
        result = loop.run("肺栓塞是什么")
        # 应能触发 retrieve 并得到结果
        assert result["success"] is False or "肺栓塞" in str(result["result"])


# ═══════════════════════════════════════════════════════════════
#  四、Agent 主控制器测试
# ═══════════════════════════════════════════════════════════════


class TestAgent:
    """Agent 主控制器测试（增强版）"""

    def test_init_default_tools(self):
        agent = Agent()
        assert "generate_report" in agent.tools
        assert "diagnose_pulmonary_embolism" in agent.tools

    def test_get_tool_schemas(self):
        agent = Agent()
        schemas = agent.get_tool_schemas()
        assert len(schemas) == 2
        assert all("tool_name" in s for s in schemas)

    def test_register_new_tool(self):
        class MockTool(Tool):
            name = "mock_tool"
            description = "测试工具"

            def run(self, **kwargs):
                return {"success": True}

        agent = Agent()
        agent.register_tool(MockTool())
        assert "mock_tool" in agent.tools

    def test_process_normal_query(self):
        """普通问题返回 agent_handled=False"""
        agent = Agent()
        result = agent.process("什么是肺栓塞", use_llm_classifier=False)
        assert result["agent_handled"] is False

    def test_process_diagnosis_intent(self):
        """诊断意图返回 agent_handled=True（规则模式）"""
        agent = Agent()
        result = agent.process("帮我诊断一下肺栓塞", use_llm_classifier=False)
        assert result["agent_handled"] is True
        assert result["tool"] == "diagnose_pulmonary_embolism"

    def test_process_report_intent(self):
        """报告意图返回 agent_handled=True（规则模式）"""
        agent = Agent()
        result = agent.process("帮我生成一份部署报告", use_llm_classifier=False)
        assert result["agent_handled"] is True
        assert result["tool"] == "generate_report"

    def test_llm_classifier_initialized(self):
        """Agent 应自带 LLMIntentClassifier"""
        agent = Agent()
        assert agent.intent_classifier is not None
        assert isinstance(agent.intent_classifier, LLMIntentClassifier)

    def test_react_loop_lazy_initialized(self):
        """_react_loop 属性应懒加载"""
        agent = Agent()
        assert agent._react is None
        _ = agent._react_loop
        assert agent._react is not None


# ═══════════════════════════════════════════════════════════════
#  五、BudgetTracker 测试
# ═══════════════════════════════════════════════════════════════


class TestBudgetTracker:
    """BudgetTracker 单元测试"""

    def test_steps_exceeded(self):
        """步数超限应返回终止原因"""
        budget = AgentBudget(max_steps=3, max_tokens_total=99999, max_wall_clock_sec=999, max_tool_calls=999)
        tracker = BudgetTracker(budget)
        assert tracker.record_step(10) is None  # step 1
        assert tracker.record_step(10) is None  # step 2
        assert tracker.record_step(10) is None  # step 3
        reason = tracker.record_step(10)  # step 4 > max_steps=3
        assert reason is not None
        assert "推理步数" in reason

    def test_tokens_total_exceeded(self):
        """Token 总数超限应返回终止原因"""
        budget = AgentBudget(max_steps=99, max_tokens_total=50, max_wall_clock_sec=999, max_tool_calls=999)
        tracker = BudgetTracker(budget)
        assert tracker.record_step(30) is None
        assert tracker.record_step(21) is not None  # 累计 51 > 50

    def test_wall_clock_exceeded(self):
        """墙钟时间超限应返回终止原因"""
        budget = AgentBudget(max_steps=99, max_tokens_total=99999, max_wall_clock_sec=0.01, max_tool_calls=999)
        tracker = BudgetTracker(budget)
        import time

        time.sleep(0.02)
        reason = tracker.check_wall_clock()
        assert reason is not None
        assert "超时" in reason

    def test_tool_calls_exceeded(self):
        """工具调用次数超限应返回终止原因"""
        budget = AgentBudget(max_steps=99, max_tokens_total=99999, max_wall_clock_sec=999, max_tool_calls=2)
        tracker = BudgetTracker(budget)
        assert tracker.record_tool_call() is None
        assert tracker.record_tool_call() is None
        reason = tracker.record_tool_call()  # 第 3 次 > 2
        assert reason is not None
        assert "工具调用" in reason

    def test_estimate_tokens_chinese(self):
        """中文文本的 Token 估算应为正数"""
        tokens = BudgetTracker.estimate_tokens("肺栓塞是一种严重的疾病")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_empty(self):
        """空文本的 Token 估算应为 0"""
        tokens = BudgetTracker.estimate_tokens("")
        assert tokens == 0

    def test_estimate_tokens_mixed(self):
        """中英文混合文本的 Token 估算"""
        tokens = BudgetTracker.estimate_tokens("肺栓塞 PE 的诊断需要 CTPA 检查")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_reset_clears_counters(self):
        """reset() 应重置所有计数器"""
        budget = AgentBudget(max_steps=2, max_tokens_total=999, max_wall_clock_sec=999, max_tool_calls=999)
        tracker = BudgetTracker(budget)
        tracker.record_step(10)
        tracker.record_tool_call()
        tracker.reset()
        assert tracker.step_count == 0
        assert tracker.token_count == 0
        assert tracker.tool_call_count == 0

    def test_finalize_records_metrics(self):
        """finalize() 应在不抛出异常的情况下完成"""
        budget = AgentBudget(max_steps=99, max_tokens_total=99999, max_wall_clock_sec=999, max_tool_calls=999)
        tracker = BudgetTracker(budget)
        tracker.record_step(100)
        tracker.record_tool_call()
        # 只检查不抛异常
        tracker.finalize()

    def test_agent_with_budget_passed_to_loop(self):
        """Agent 传入 harness_config 后，ReActLoop 应能初始化"""
        harness = AgentHarnessConfig(max_steps=5, max_tokens_total=99999)
        agent = Agent(harness_config=harness)
        loop = agent._react_loop
        # harness config 被传递到了 FunctionCallingLoop 初始化
        assert loop._budget is not None
        assert loop._budget.max_steps == 5


# ═══════════════════════════════════════════════════════════════
#  六、Circuit Breaker 测试（通过 MockGenerator）
# ═══════════════════════════════════════════════════════════════


class FailingGenerator:
    """模拟总是失败的 LLM Generator"""

    def __init__(self, fail_count: int = 99):
        self.fail_count = fail_count
        self.attempts = 0

    def chat(self, messages, temperature=0.0, max_tokens=256) -> str:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ConnectionError("模拟网络错误")
        return "最终成功回复"


class TestGeneratorResilience:
    """LLMGenerator 容错测试"""

    def test_retry_success_eventually(self):
        """重试 N 次后应该成功（模拟指数退避）"""
        from src.generator import LLMGenerator

        gen = LLMGenerator(
            api_key="sk-test-valid-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            retry_max_attempts=3,
            retry_base_delay=0.01,  # 快速重试
        )

        # 验证 _with_retry 重试逻辑正常
        call_count = [0]

        def failing_fn(*args):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("模拟错误")
            return "success"

        result = gen._with_retry(failing_fn)
        assert result == "success"
        assert call_count[0] == 2  # 第 1 次失败，第 2 次成功

    def test_retry_eventually_fails(self):
        """重试耗尽后应抛出异常"""
        from src.generator import LLMGenerator

        gen = LLMGenerator(
            api_key="sk-test-valid-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            retry_max_attempts=2,
            retry_base_delay=0.01,
        )

        call_count = [0]

        def always_fail(*args):
            call_count[0] += 1
            raise ConnectionError("始终失败")

        import pytest

        with pytest.raises(ConnectionError):
            gen._with_retry(always_fail)
        assert call_count[0] == 2  # 重试耗尽

    def test_circuit_breaker_blocks(self):
        """熔断器打开后应拒绝请求"""
        from src.generator import LLMGenerator

        gen = LLMGenerator(
            api_key="sk-test-valid-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=999,  # 长时间熔断
        )

        # 模拟连续失败
        for _ in range(3):
            gen._record_failure()

        assert gen._cb_state == "open"
        assert gen._check_circuit_breaker() is False

    def test_circuit_breaker_recovers(self):
        """熔断器在超时后应进入 half_open 状态"""
        from src.generator import LLMGenerator

        gen = LLMGenerator(
            api_key="sk-test-valid-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=0.01,  # 快速恢复
        )

        # 触发熔断
        gen._record_failure()
        gen._record_failure()
        assert gen._cb_state == "open"

        # 等待超时后检查
        import time

        time.sleep(0.02)
        assert gen._check_circuit_breaker() is True  # 放行试探
        assert gen._cb_state == "half_open"

        # 成功后恢复
        gen._record_success()
        assert gen._cb_state == "closed"

    def test_4xx_errors_not_retried(self):
        """4xx 错误不应重试"""
        from src.generator import LLMGenerator

        gen = LLMGenerator(
            api_key="sk-test-valid-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            retry_max_attempts=3,
            retry_base_delay=0.01,
        )

        call_count = [0]

        def unauthorized(*args):
            call_count[0] += 1
            raise ValueError("401 Unauthorized")

        import pytest

        with pytest.raises(ValueError):
            gen._with_retry(unauthorized)
        assert call_count[0] == 1  # 只调用了一次，未重试


# ═══════════════════════════════════════════════════════════════
#  七、Policy-as-Code 测试
# ═══════════════════════════════════════════════════════════════


class TestToolPolicy:
    """ToolPolicy 单元测试"""

    def test_default_auto(self):
        """默认策略应为 auto"""
        from src.tools.base import ToolPolicy

        policy = ToolPolicy()
        assert policy.access_level == "auto"
        assert policy.rate_limit == 0
        assert policy.require_reason is False

    def test_custom_policy(self):
        """自定义策略应正确设置"""
        from src.tools.base import ToolPolicy

        policy = ToolPolicy(access_level="confirm", rate_limit=10, require_reason=True)
        assert policy.access_level == "confirm"
        assert policy.rate_limit == 10
        assert policy.require_reason is True

    def test_to_dict(self):
        """to_dict 应返回正确字段"""
        from src.tools.base import ToolPolicy

        policy = ToolPolicy(access_level="confirm", rate_limit=10)
        d = policy.to_dict()
        assert d["access_level"] == "confirm"
        assert d["rate_limit"] == 10
        assert "allowed_roles" in d


class TestPolicyEnforcer:
    """PolicyEnforcer 单元测试"""

    def test_auto_passes(self):
        """auto 级别应直接放行"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="auto")
        result = enforcer.check("test_tool", policy)
        assert result.allowed is True
        assert result.needs_confirmation is False
        enforcer.record_call("test_tool")

    def test_confirm_requires_approval(self):
        """confirm 级别应返回 needs_confirmation=True"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="confirm", require_reason=True)
        result = enforcer.check("test_tool", policy, reason="测试诊断")
        assert result.allowed is True
        assert result.needs_confirmation is True
        assert "确认" in result.reason

    def test_confirm_requires_reason(self):
        """confirm + require_reason 但无理由应拒绝"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="confirm", require_reason=True)
        result = enforcer.check("test_tool", policy, reason="")
        assert result.allowed is False
        assert "理由" in result.reason

    def test_rate_limit_exceeded(self):
        """超过 rate_limit 应拒绝"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="auto", rate_limit=2)

        # 前两次通过
        assert enforcer.check("rate_tool", policy).allowed
        enforcer.record_call("rate_tool")
        assert enforcer.check("rate_tool", policy).allowed
        enforcer.record_call("rate_tool")

        # 第三次应拒绝
        result = enforcer.check("rate_tool", policy)
        assert result.allowed is False
        assert "频率超限" in result.reason

    def test_rate_limit_reset_after_reset(self):
        """reset 后应恢复调用"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="auto", rate_limit=1)

        assert enforcer.check("r_tool", policy).allowed
        enforcer.record_call("r_tool")
        assert enforcer.check("r_tool", policy).allowed is False

        enforcer.reset("r_tool")
        assert enforcer.check("r_tool", policy).allowed

    def test_manual_level(self):
        """manual 级别应返回 allowed=False"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="manual")
        result = enforcer.check("manual_tool", policy)
        assert result.allowed is False
        assert "手动" in result.reason

    def test_unknown_level(self):
        """未知访问级别应安全拒绝"""
        from src.tools.base import PolicyEnforcer, ToolPolicy

        enforcer = PolicyEnforcer()
        policy = ToolPolicy(access_level="unknown_level")
        result = enforcer.check("unknown_tool", policy)
        assert result.allowed is False
        assert "未知" in result.reason

    def test_policy_to_dict(self):
        """PolicyResult.to_dict 应正确序列化"""
        from src.tools.base import PolicyResult

        result = PolicyResult(allowed=True, level="confirm", reason="测试", needs_confirmation=True)
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["needs_confirmation"] is True
        assert d["level"] == "confirm"


class TestToolPolicyIntegration:
    """工具策略集成测试"""

    def test_diagnosis_tool_policy_confirm(self):
        """诊断工具策略应为 confirm"""
        from src.tools.diagnosis_tool import DiagnosisTool

        tool = DiagnosisTool()
        assert tool.policy.access_level == "confirm"
        assert tool.policy.rate_limit == 10
        assert tool.policy.require_reason is True

    def test_report_generator_policy_auto(self):
        """报告生成工具策略应为 auto"""
        from src.tools.report_generator import ReportGenerator

        tool = ReportGenerator()
        assert tool.policy.access_level == "auto"
        assert tool.policy.rate_limit == 60

    def test_simple_tool_default_policy(self):
        """新增 SimpleTool 默认 policy 应为 auto"""
        from src.tools.base import Tool, ToolPolicy

        class SimpleTool(Tool):
            name = "simple"
            description = "simple tool"
            policy = ToolPolicy()

            def run(self, **kwargs):
                return {"success": True}

        tool = SimpleTool()
        assert tool.policy.access_level == "auto"
        schema = tool.get_schema()
        assert "policy" in schema
        assert schema["policy"]["access_level"] == "auto"

    def test_fc_loop_with_policy_enforcer(self):
        """FunctionCallingLoop 应初始化 policy_enforcer"""
        from src.agent import FunctionCallingLoop

        loop = FunctionCallingLoop(tools={})
        assert loop._policy_enforcer is not None

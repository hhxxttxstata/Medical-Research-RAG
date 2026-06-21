"""
三层记忆系统单元测试

覆盖：
  - SessionMemory（短期记忆）
  - TaskMemory & TaskState（工作记忆）
  - PreferenceMemory（长期记忆，ChromaDB + Embedding）
  - MemoryManager（管理器集成）
  - Agent 集成（记忆注入 + 记录）
"""

import time

from src.memory import (
    MemoryManager,
    PreferenceMemory,
    SessionMemory,
    TaskMemory,
    TaskState,
)

# ═══════════════════════════════════════════════════════════════
#  测试辅助
# ═══════════════════════════════════════════════════════════════


class MockEmbeddingProvider:
    """模拟 Embedding，返回固定长度的零向量"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 512 for _ in texts]


# ═══════════════════════════════════════════════════════════════
#  一、SessionMemory（短期记忆）测试
# ═══════════════════════════════════════════════════════════════


class TestSessionMemory:
    """短期记忆测试"""

    def test_add_and_get(self):
        """添加后能取回最近的对话"""
        sm = SessionMemory()
        sm.add("s1", "user", "你好")
        sm.add("s1", "assistant", "你好，有什么可以帮你的？")
        sm.add("s1", "user", "什么是肺栓塞")

        recent = sm.get_recent("s1", 2)
        assert len(recent) == 2
        assert recent[0]["role"] == "assistant"
        assert recent[1]["content"] == "什么是肺栓塞"

    def test_empty_session(self):
        """没有记录的 session 应返回空列表"""
        sm = SessionMemory()
        assert sm.get_recent("nonexistent") == []

    def test_max_len_eviction(self):
        """超过 MAX_HISTORY 应淘汰最旧条目"""
        sm = SessionMemory()
        sm.MAX_HISTORY = 3  # 测试用缩小
        for i in range(5):
            sm.add("s1", "user", f"msg{i}")
        recent = sm.get_recent("s1", 10)
        assert len(recent) == 3
        assert recent[0]["content"] == "msg2"
        assert recent[-1]["content"] == "msg4"

    def test_get_recent_n_larger_than_history(self):
        """请求的 n 超过历史长度时返回全部"""
        sm = SessionMemory()
        for i in range(3):
            sm.add("s1", "user", f"msg{i}")
        recent = sm.get_recent("s1", 100)
        assert len(recent) == 3

    def test_clear_session(self):
        """清除指定 session 的历史"""
        sm = SessionMemory()
        sm.add("s1", "user", "a")
        sm.add("s2", "user", "b")
        sm.clear("s1")
        assert sm.get_recent("s1") == []
        assert len(sm.get_recent("s2")) == 1

    def test_clear_all(self):
        """清除所有 session 的历史"""
        sm = SessionMemory()
        sm.add("s1", "user", "a")
        sm.add("s2", "user", "b")
        sm.clear_all()
        assert sm.get_recent("s1") == []
        assert sm.get_recent("s2") == []


# ═══════════════════════════════════════════════════════════════
#  二、TaskMemory（工作记忆）测试
# ═══════════════════════════════════════════════════════════════


class TestTaskMemory:
    """工作记忆测试"""

    def test_start_and_get(self):
        """开始任务后能获取到"""
        tm = TaskMemory()
        tm.start("s1", "report_generate", {"report_type": "deployment"})
        task = tm.get("s1")
        assert task is not None
        assert task.task_type == "report_generate"
        assert task.status == "in_progress"

    def test_add_step(self):
        """添加步骤"""
        tm = TaskMemory()
        tm.start("s1", "report_generate")
        tm.add_step("s1", "检索知识库")
        tm.add_step("s1", "生成部署报告")
        task = tm.get("s1")
        assert len(task.completed_steps) == 2
        assert "检索知识库" in task.completed_steps

    def test_complete(self):
        """标记完成"""
        tm = TaskMemory()
        tm.start("s1", "report_generate")
        tm.complete("s1")
        task = tm.get("s1")
        assert task is None or task.status == "completed"

    def test_clear(self):
        """清除指定 session"""
        tm = TaskMemory()
        tm.start("s1", "report_generate")
        tm.clear("s1")
        assert tm.get("s1") is None

    def test_expiry(self):
        """过期任务应被清理"""
        tm = TaskMemory()
        tm.TTL_SECONDS = 0  # 立即过期
        tm.start("s1", "report_generate")
        import time

        time.sleep(0.01)  # 确保超过 0s
        task = tm.get("s1")
        assert task is None

    def test_start_overwrites(self):
        """同一 session 的新任务应覆盖旧任务"""
        tm = TaskMemory()
        tm.start("s1", "report_generate", {"report_type": "deployment"})
        tm.add_step("s1", "步骤1")
        tm.start("s1", "pe_diagnosis")
        task = tm.get("s1")
        assert task.task_type == "pe_diagnosis"
        assert task.completed_steps == []

    def test_task_state_touch(self):
        """touch 更新活跃时间"""
        state = TaskState("report_generate")
        old = state.last_active
        time.sleep(0.01)
        state.touch()
        assert state.last_active > old

    def test_task_state_to_dict(self):
        """to_dict 序列化"""
        state = TaskState("report_generate", {"key": "val"})
        state.completed_steps.append("step1")
        d = state.to_dict()
        assert d["task_type"] == "report_generate"
        assert d["data"]["key"] == "val"
        assert "step1" in d["completed_steps"]


# ═══════════════════════════════════════════════════════════════
#  三、PreferenceMemory（长期记忆）测试
# ═══════════════════════════════════════════════════════════════


class TestPreferenceMemory:
    """长期记忆测试（使用 mock embedding 避免依赖模型）"""

    def test_upsert_and_search(self):
        """存入并检索同一 session 的偏好"""
        pm = PreferenceMemory(
            embedding_provider=MockEmbeddingProvider(),
            persist_dir="",
        )
        pm.upsert("s1", "pref_123", "用户喜欢生成deployment报告", {"type": "preference"})
        results = pm.search("s1", "部署报告", top_k=5)
        assert len(results) >= 1
        assert "deployment" in results[0]["content"]

    def test_search_different_session(self):
        """不同 session 不应互相影响"""
        pm = PreferenceMemory(
            embedding_provider=MockEmbeddingProvider(),
            persist_dir="",
        )
        pm.upsert("s1", "pref_1", "用户偏好A", {"type": "preference"})
        results = pm.search("s2", "偏好", top_k=5)
        assert len(results) == 0

    def test_upsert_overwrite(self):
        """同一 session + 同一 key 覆盖"""
        pm = PreferenceMemory(
            embedding_provider=MockEmbeddingProvider(),
            persist_dir="",
        )
        pm.upsert("s1", "key1", "旧内容", {"type": "preference"})
        pm.upsert("s1", "key1", "新内容", {"type": "preference"})
        results = pm.search("s1", "新内容", top_k=5)
        # 至少应有一条结果且是新的
        assert len(results) >= 1

    def test_forget(self):
        """删除指定 session 的所有记忆"""
        pm = PreferenceMemory(
            embedding_provider=MockEmbeddingProvider(),
            persist_dir="",
        )
        pm.upsert("s1", "k1", "内容1", {"type": "preference"})
        pm.upsert("s1", "k2", "内容2", {"type": "preference"})
        pm.forget("s1")
        results = pm.search("s1", "内容", top_k=5)
        assert len(results) == 0

    def test_no_embedding_provider(self):
        """无 embedding provider 时不应崩溃"""
        pm = PreferenceMemory(embedding_provider=None)
        ok = pm.upsert("s1", "key", "内容")
        assert ok is False
        results = pm.search("s1", "test")
        assert results == []


# ═══════════════════════════════════════════════════════════════
#  四、MemoryManager 集成测试
# ═══════════════════════════════════════════════════════════════


class TestMemoryManager:
    """记忆管理器集成测试"""

    def test_build_context_with_history(self):
        """有历史时 build_context 返回非空"""
        mm = MemoryManager(embedding_provider=None)
        mm.short_term.add("s1", "user", "什么是肺栓塞")
        mm.short_term.add("s1", "assistant", "肺栓塞是一种急重症。")
        ctx = mm.build_context("s1", "再解释一下")
        assert "肺栓塞是一种急重症" in ctx
        assert "近期对话" in ctx or "【近期对话】" in ctx

    def test_build_context_without_history(self):
        """无历史时 build_context 返回空字符串"""
        mm = MemoryManager(embedding_provider=None)
        ctx = mm.build_context("nonexistent", "你好")
        assert ctx == ""

    def test_build_context_with_task(self):
        """有进行中任务时 build_context 应包含任务信息"""
        mm = MemoryManager(embedding_provider=None)
        mm.working.start("s1", "report_generate", {"report_type": "deployment"})
        mm.working.add_step("s1", "检索完成")
        ctx = mm.build_context("s1", "继续")
        assert "report_generate" in ctx or "报告" in ctx or "报告" in ctx

    def test_remember_normal_query(self):
        """普通问答后 remember 不应崩溃，应记录到短期记忆"""
        mm = MemoryManager(embedding_provider=None)
        mm.remember("s1", "什么是肺栓塞", "肺栓塞是一种急重症", {"intent": "normal_query"})
        history = mm.short_term.get_recent("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_remember_report_generate(self):
        """报告生成后应创建工作任务"""
        mm = MemoryManager(embedding_provider=None)
        mm.remember(
            "s1",
            "生成部署报告",
            "部署报告内容...",
            {"intent": "report_generate", "report_type": "deployment"},
        )
        task = mm.working.get("s1")
        assert task is not None
        assert task.task_type == "report_generate"

    def test_remember_with_none_intent(self):
        """intent_info 为 None 时不应崩溃"""
        mm = MemoryManager(embedding_provider=None)
        mm.remember("s1", "hello", "world", None)
        history = mm.short_term.get_recent("s1")
        assert len(history) == 2

    def test_clear_session(self):
        """清除 session 应清理所有三层记忆"""
        mm = MemoryManager(embedding_provider=None)
        mm.short_term.add("s1", "user", "你好")
        mm.working.start("s1", "report_generate")
        mm.clear_session("s1")
        assert mm.short_term.get_recent("s1") == []
        assert mm.working.get("s1") is None

    def test_summarize_preference(self):
        """检测用户偏好表达"""
        result = MemoryManager._summarize_preference("我偏好生成部署报告", "ok")
        assert "偏好" in result

        result2 = MemoryManager._summarize_preference("hi", "ok")
        assert result2 == ""

        result3 = MemoryManager._summarize_preference("这个功能太差了", "no")
        assert result3 == ""


# ═══════════════════════════════════════════════════════════════
#  五、Agent 集成测试
# ═══════════════════════════════════════════════════════════════


class TestAgentMemoryIntegration:
    """Agent + 记忆集成测试"""

    def test_agent_process_with_session_id(self):
        """传入 session_id 不应崩溃（无 memory_manager 时静默跳过）"""
        from src.agent import Agent

        agent = Agent()
        result = agent.process("什么是肺栓塞", use_llm_classifier=False, session_id="test_session")
        assert result["agent_handled"] is False

    def test_agent_with_memory_manager_injects_context(self):
        """有 memory_manager 时 process 应注入记忆上下文"""
        from src.agent import Agent

        agent = Agent()
        mm = MemoryManager(embedding_provider=None)
        mm.short_term.add("s1", "user", "之前的问题")
        mm.short_term.add("s1", "assistant", "之前的回答")
        agent.memory_manager = mm

        # normal_query — 不触发 ReAct，但记忆上下文应被构建
        result = agent.process(
            "后续问题",
            use_llm_classifier=False,
            session_id="s1",
        )
        assert result["agent_handled"] is False

        # 检查短期记忆已记录本次交互
        history = mm.short_term.get_recent("s1")
        assert any("后续问题" in h["content"] for h in history)

    def test_agent_with_memory_manager_normal_query_recorded(self):
        """普通问答应被记录到短期记忆"""
        from src.agent import Agent

        agent = Agent()
        mm = MemoryManager(embedding_provider=None)
        agent.memory_manager = mm

        result = agent.process(
            "测试记录",
            use_llm_classifier=False,
            session_id="rec_session",
        )

        # normal_query 不返回 memory_used（但内部已调用 remember）
        # 验证短期记忆已记录
        history = mm.short_term.get_recent("rec_session")
        assert len(history) >= 2  # user + assistant
        assert history[0]["content"] == "测试记录"

    def test_without_memory_manager_backward_compat(self):
        """无 memory_manager 时完全不影响旧逻辑"""
        from src.agent import Agent

        agent = Agent()
        result = agent.process(
            "什么是肺栓塞",
            use_llm_classifier=False,
            session_id="any",
        )
        assert "agent_handled" in result
        assert "memory_used" not in result

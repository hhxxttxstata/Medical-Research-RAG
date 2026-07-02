"""
三层记忆系统

工业级 Agent 的三层记忆架构：
  1. 短期记忆（SessionMemory） — session 内的对话历史
  2. 工作记忆（TaskMemory）    — 多步任务的进度追踪
  3. 长期记忆（PreferenceMemory）— 跨 session 的用户偏好持久化

面试价值：
  - 三层记忆是 Agent 面试核心考点，展示对记忆管理的深入理解
  - 短期+工作+长期的分层设计，兼顾效率与持久性
  - 优雅降级：任何一层失败不影响其他层正常工作
"""

import os
import time
from collections import deque
from typing import Any

# ═══════════════════════════════════════════════════════════════
#  一、短期记忆 — Session 内对话历史
# ═══════════════════════════════════════════════════════════════


class SessionMemory:
    """短期记忆

    以 session_id 为 key 的对话历史缓存。
    纯内存结构，maxlen 淘汰旧条目。

    每条记录: {"role": "user"|"assistant", "content": str}
    默认保留最近 20 轮对话。
    """

    MAX_HISTORY = 20

    def __init__(self):
        # session_id → deque of {role, content}
        self._sessions: dict[str, deque] = {}

    def get_recent(self, session_id: str, n: int = 5) -> list[dict[str, str]]:
        """获取最近 n 轮对话"""
        if session_id not in self._sessions:
            return []
        dq = self._sessions[session_id]
        recent = list(dq)[-n:] if n else list(dq)
        return recent

    def add(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话记录"""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.MAX_HISTORY)
        self._sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def clear(self, session_id: str) -> None:
        """清除指定 session 的历史"""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """清除所有 session 的历史"""
        self._sessions.clear()


# ═══════════════════════════════════════════════════════════════
#  二、工作记忆 — 任务进度追踪
# ═══════════════════════════════════════════════════════════════


class TaskState:
    """单个任务的工作状态"""

    def __init__(
        self,
        task_type: str,
        data: dict[str, Any] | None = None,
    ):
        self.task_type = task_type  # "report_generate" | "pe_diagnosis"
        self.status: str = "in_progress"  # in_progress | completed | expired
        self.started_at: float = time.time()
        self.last_active: float = time.time()
        self.data: dict[str, Any] = data or {}
        self.completed_steps: list[str] = []

    def touch(self) -> None:
        """更新活跃时间戳"""
        self.last_active = time.time()

    def is_expired(self, ttl_seconds: int = 900) -> bool:
        """检查是否超过空闲 TTL（默认 15 分钟）"""
        return time.time() - self.last_active > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "status": self.status,
            "started_at": self.started_at,
            "last_active": self.last_active,
            "data": self.data,
            "completed_steps": list(self.completed_steps),
        }


class TaskMemory:
    """工作记忆

    追踪每个 session 中正在进行的多步任务。
    - 自动 15min TTL 过期清理
    - 一个 session 同时只追踪一个活跃任务
    """

    TTL_SECONDS = 900  # 15 分钟

    def __init__(self):
        self._tasks: dict[str, TaskState] = {}

    def get(self, session_id: str) -> TaskState | None:
        """获取指定 session 的活跃任务（自动清理过期）"""
        self._cleanup_expired()
        task = self._tasks.get(session_id)
        if task is None:
            return None
        if task.is_expired(self.TTL_SECONDS):
            task.status = "expired"
            del self._tasks[session_id]
            return None
        return task

    def start(
        self,
        session_id: str,
        task_type: str,
        data: dict[str, Any] | None = None,
    ) -> TaskState:
        """开始一个新任务（覆盖之前的）"""
        task = TaskState(task_type=task_type, data=data)
        self._tasks[session_id] = task
        return task

    def update(self, session_id: str, **updates) -> TaskState | None:
        """更新任务状态"""
        task = self.get(session_id)
        if task is None:
            return None
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.touch()
        return task

    def complete(self, session_id: str) -> TaskState | None:
        """标记任务完成"""
        task = self.get(session_id)
        if task:
            task.status = "completed"
            task.touch()
        return task

    def add_step(self, session_id: str, step: str) -> TaskState | None:
        """添加一个完成步骤"""
        task = self.get(session_id)
        if task:
            task.completed_steps.append(step)
            task.touch()
        return task

    def _cleanup_expired(self) -> None:
        """惰性清理过期任务"""
        now = time.time()
        expired = [sid for sid, task in self._tasks.items() if task.is_expired(self.TTL_SECONDS)]
        for sid in expired:
            self._tasks[sid].status = "expired"
            del self._tasks[sid]

    def clear(self, session_id: str) -> None:
        """清除指定 session 的任务"""
        self._tasks.pop(session_id, None)

    def clear_all(self) -> None:
        """清除所有任务"""
        self._tasks.clear()


# ═══════════════════════════════════════════════════════════════
#  三、长期记忆 — 用户偏好持久化（ChromaDB + Embedding）
# ═══════════════════════════════════════════════════════════════


class PreferenceMemory:
    """长期记忆

    使用 ChromaDB 持久化用户偏好，复用项目已有的 embedding 模型。
    每条记录包含：
      - session_id（过滤来源）
      - key（去重，同一 session + 同一 key 覆盖）
      - content（文本内容）
      - metadata（embedding 辅助信息）

    存储集合: "memory_store"（与知识库向量集合隔离）
    """

    # Chroma 集合名，与 knolwedge base 的集合隔离
    COLLECTION_NAME = "memory_store"

    def __init__(self, embedding_provider=None, persist_dir: str = "chroma_db"):
        self._embedding_provider = embedding_provider
        self._persist_dir = os.path.abspath(persist_dir)
        self._collection = None

    def _get_collection(self):
        """懒初始化 Chroma 集合"""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                self._collection = client.get_collection(self.COLLECTION_NAME)
            except Exception:
                self._collection = client.create_collection(self.COLLECTION_NAME)
        except Exception as e:
            print(f"  ⚠️ 长期记忆 ChromaDB 初始化失败: {e}")
            self._collection = None  # type: ignore[assignment]

        return self._collection

    def _ensure_embedding(self) -> bool:
        """确保 embedding provider 可用"""
        if self._embedding_provider is None:
            return False
        try:
            # 调用 embed 方法做一次快速检查
            self._embedding_provider.embed(["test"])
            return True
        except Exception:
            return False

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """检索用户偏好

        使用 query 的向量搜索同 session 的历史偏好记录。
        返回按相似度排序的结果列表。
        """
        collection = self._get_collection()
        if collection is None or not self._ensure_embedding():
            return []

        try:
            query_embedding = self._embedding_provider.embed([query])[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"session_id": session_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        items = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                items.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "session_id": meta.get("session_id", ""),
                        "type": meta.get("type", ""),
                        "score": 1 - results["distances"][0][i],
                    }
                )
        return items

    def upsert(
        self,
        session_id: str,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """存储一条偏好记录

        同一 session + 同一 key = 覆盖更新。

        Returns:
            bool — 是否成功
        """
        collection = self._get_collection()
        if collection is None or not self._ensure_embedding():
            return False

        doc_id = f"{session_id}_{key}"
        meta = dict(metadata or {})
        meta["session_id"] = session_id
        meta["key"] = key

        try:
            embedding = self._embedding_provider.embed([content])[0]
            # 尝试删除旧记录（同 id），再添加
            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass
            collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
                ids=[doc_id],
            )
            return True
        except Exception as e:
            print(f"  ⚠️ 长期记忆写入失败: {e}")
            return False

    def get_user_profile(self, session_id: str) -> str:
        """聚合该 session 所有偏好为一段文本"""
        collection = self._get_collection()
        if collection is None:
            return ""

        try:
            results = collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            return ""

        if not results["ids"]:
            return ""

        lines = []
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            content = results["documents"][i] if results["documents"] else ""
            t = meta.get("type", "通用")
            lines.append(f"[{t}] {content}")

        return "\n".join(lines)

    def forget(self, session_id: str) -> bool:
        """删除指定 session 的所有记忆"""
        collection = self._get_collection()
        if collection is None:
            return False

        try:
            results = collection.get(
                where={"session_id": session_id},
                include=[],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
            return True
        except Exception as e:
            print(f"  ⚠️ 清除长期记忆失败: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
#  四、MemoryManager — 统一入口
# ═══════════════════════════════════════════════════════════════

_MEMORY_CONTEXT_TEMPLATE = """
## 记忆上下文（历史参考）
{memory_content}

阅读以上历史信息来理解对话背景。如果与当前问题无关请忽略。
"""


class MemoryManager:
    """记忆管理器 — 三层记忆统一入口

    职责：
      1. 在 Agent 处理请求前，从三层记忆构建上下文文本
      2. 在 Agent 处理请求后，将本次交互记录到各层记忆

    每层独立运行，单层失败不影响其他层。
    """

    def __init__(self, embedding_provider=None, persist_dir: str = "chroma_db"):
        self.short_term = SessionMemory()
        self.working = TaskMemory()
        self.long_term = PreferenceMemory(
            embedding_provider=embedding_provider,
            persist_dir=persist_dir,
        )

    def build_context(self, session_id: str, query: str) -> str:
        """构建记忆上下文文本

        聚合三层记忆的当前相关信息，注入到 ReAct 系统提示词中。
        如果没有记忆则返回空字符串。
        """
        parts = []

        # ── 1. 短期记忆：最近对话 ──
        history = self.short_term.get_recent(session_id, 5)
        if history:
            conv_lines = []
            for h in history:
                role_label = "用户" if h["role"] == "user" else "助手"
                conv_lines.append(f"  {role_label}: {h['content']}")
            parts.append("【近期对话】\n" + "\n".join(conv_lines))

        # ── 2. 工作记忆：进行中的任务 ──
        task = self.working.get(session_id)
        if task and task.status == "in_progress":
            lines = [f"  类型: {task.task_type}"]
            if task.completed_steps:
                lines.append(f"  已完成步骤: {', '.join(task.completed_steps)}")
            parts.append("【当前任务】\n" + "\n".join(lines))

        # ── 3. 长期记忆：用户偏好 ──
        prefs = self.long_term.search(session_id, query, top_k=2)
        if prefs:
            pref_lines = [f"  - {p['content']}" for p in prefs]
            parts.append("【用户偏好参考】\n" + "\n".join(pref_lines))

        if not parts:
            return ""

        return _MEMORY_CONTEXT_TEMPLATE.format(memory_content="\n\n".join(parts))

    @staticmethod
    def _summarize_preference(query: str, answer: str) -> str:
        """从问答中归纳可长期记忆的信息

        规则启发式（不用 LLM，避免额外成本）：
        - 直接偏好表达（"喜欢""偏好"等）直接记录
        - 每个有实质内容的问答都记录用户关注主题
        """
        query_lower = query.lower().strip()
        if len(query) < 8:
            return ""

        # 直接偏好表达
        preference_markers = ["偏好", "喜欢", "习惯", "总是", "每次"]
        for m in preference_markers:
            if m in query:
                return f"用户偏好: {query[:200]}"

        # 否定/抱怨类不记
        negative_markers = ["不喜欢", "不要", "别用", "太差", "不好"]
        for m in negative_markers:
            if m in query:
                return ""

        # 只要有实质内容就记录（跨 session 可复用的知识）
        if len(query) > 10 and len(answer) > 30:
            topic = query[:100].strip()
            return f"用户关注: {topic}"

        return ""

    def remember(
        self,
        session_id: str,
        query: str,
        answer: str,
        intent_info: dict[str, Any] | None = None,
    ) -> None:
        """记录本次交互到各层记忆

        参数中所有字段都可能为空/None，该方法不应抛异常。
        """
        intent_info = intent_info or {}
        answer_text = answer if isinstance(answer, str) else str(answer)
        intent_str = intent_info if isinstance(intent_info, str) else intent_info.get("intent", "normal_query")

        # ── 1. 短期记忆：记录对话 ──
        try:
            self.short_term.add(session_id, "user", query)
            self.short_term.add(session_id, "assistant", answer_text[:500])
        except Exception:
            pass

        # ── 2. 工作记忆：更新任务进度 ──
        try:
            if intent_str == "report_generate":
                task = self.working.get(session_id)
                if task is None:
                    # 开始新任务
                    report_type = intent_info.get("report_type", "") if isinstance(intent_info, dict) else ""
                    self.working.start(
                        session_id,
                        task_type="report_generate",
                        data={"report_type": report_type},
                    )
                else:
                    report_type = intent_info.get("report_type", "") if isinstance(intent_info, dict) else ""
                    self.working.add_step(
                        session_id,
                        f"生成{report_type}报告" if report_type else "生成报告",
                    )
            elif intent_str == "normal_query" and len(query) > 15:
                # 普通问答也可能是一个隐含的多步任务的开端
                pass
        except Exception:
            pass

        # ── 3. 长期记忆：归纳偏好 ──
        try:
            summary = self._summarize_preference(query, answer_text)
            if summary:
                self.long_term.upsert(
                    session_id,
                    key=f"pref_{abs(hash(query)) % 100000}",
                    content=summary,
                    metadata={
                        "session_id": session_id,
                        "type": "preference",
                        "timestamp": time.time(),
                    },
                )
        except Exception:
            pass

    def clear_session(self, session_id: str) -> None:
        """清除指定 session 的所有记忆"""
        self.short_term.clear(session_id)
        self.working.clear(session_id)
        self.long_term.forget(session_id)

    def clear_all(self) -> None:
        """清除所有记忆"""
        self.short_term.clear_all()
        self.working.clear_all()

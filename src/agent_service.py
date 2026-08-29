"""
Agentic Query Service — 研究层 → 服务层的薄适配

把冻结的 Agentic RAG v2.1（Cost-aware Policy + LangGraph runtime，Step 16
parity + Step 14 cost ablation 已验证）作为可调用服务暴露：
POST /query → LangGraph Agentic RAG（v2.1）→ 结构化响应。

设计原则（与 README「研究层/服务层分离」一致）：
  - 复用 RAGPipeline 的 retriever（同一 Milvus 连接——Milvus Lite 双连接会
    产生污染数据，评测纪律：串行独占）
  - reranker / generator / agent 懒加载（不阻塞服务启动；首次调用 ~10-20s）
  - threading.Lock 串行化 agentic 查询（Milvus Lite 单进程独占纪律）
  - grader_called 由 cost-aware 门控如实上报（cheap signal 直接决策时不调
    grader；LLM 不可用时自动走规则 fallback）

响应契约:
    {
        "question": str,
        "answer": str,                    # ACCEPT 生成回答 / ABSTAIN 拒答文本
        "status": "ACCEPT" | "ABSTAIN",
        "evidence": [{id, filename, score, text}],   # final_evidence（≤10 条）
        "route": ["RETRIEVE", ...],       # 决策序列
        "iterations": int,
        "grader_called": bool,            # 本轮是否调用了 LLM grader
        "latency_ms": int,
        "abstain_reason": str | None      # 仅 ABSTAIN 时有值
    }
"""

import threading
import time
from typing import Any

from .cost_aware_agentic_rag import CostAwareAgenticRAG
from .generator import create_generator
from .langgraph_agent import LangGraphAgenticRAG
from .reranker import CrossEncoderReranker


class AgentQueryService:
    """薄封装：pipeline.retriever + reranker + generator → LangGraph Agentic RAG v2.1"""

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._lock = threading.Lock()  # Milvus Lite 串行独占
        self._agent: LangGraphAgenticRAG | None = None
        self._generator = None
        self._init_error: str | None = None

    # ── 懒加载 ──────────────────────────────────────

    def _ensure_agent(self):
        if self._agent is not None:
            return
        if self._init_error:
            raise RuntimeError(f"Agentic 服务初始化失败: {self._init_error}")
        try:
            reranker = CrossEncoderReranker()
            reranker._load_model()
            generator = create_generator()
            agent = CostAwareAgenticRAG(
                retriever=self._pipeline.retriever,  # 复用服务检索器（同一 Milvus 连接）
                generator=generator,
                reranker=reranker,
                max_iterations=2,
            )
            self._agent = LangGraphAgenticRAG(agent)
            self._generator = generator
        except Exception as e:  # pragma: no cover
            self._init_error = str(e)
            raise

    @property
    def ready(self) -> bool:
        return self._agent is not None

    # ── 查询入口 ────────────────────────────────────

    def query(self, question: str, fetch_k: int = 20) -> dict[str, Any]:
        """执行一次 Agentic 查询（同步阻塞，线程池调用方负责不卡事件循环）"""
        with self._lock:
            self._ensure_agent()
            assert self._agent is not None  # _ensure_agent 失败会抛异常
            t0 = time.monotonic()
            result = self._agent.run(question, fetch_k=fetch_k)
            latency_ms = int((time.monotonic() - t0) * 1000)

            state = result.get("state")
            abstained = bool(result.get("abstained"))
            evidence = []
            for s in result.get("sources", [])[:10]:
                meta = s.get("metadata") or {}
                evidence.append(
                    {
                        "id": s.get("id", ""),
                        "filename": meta.get("filename", ""),
                        "score": round(float(s.get("_rerank_score", s.get("score", 0.0))), 4),
                        "text": (s.get("text") or "")[:200],
                    }
                )

            return {
                "question": question,
                "answer": result.get("answer", ""),
                "status": "ABSTAIN" if abstained else "ACCEPT",
                "evidence": evidence,
                "route": result.get("route", []),
                "iterations": result.get("iterations", 0),
                "grader_called": bool(result.get("grader_called", False)),
                "latency_ms": latency_ms,
                "abstain_reason": getattr(state, "abstain_reason", "") if abstained else None,
            }

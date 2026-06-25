"""
Metrics — Prometheus 指标层

提供：
  - Counter/Histogram/Gauge 定义
  - record_* 便捷函数
  - get_metrics_registry() 获取注册表供 /metrics 端点使用

指标清单：
  REQUESTS_TOTAL     Counter   — 请求总量（endpoint, status）
  REQUEST_LATENCY    Histogram — 请求延迟分布（endpoint）
  RETRIEVAL_SCORE    Histogram — 检索分数分布
  LLM_LATENCY        Histogram — LLM 调用延迟
  AGENT_STEPS        Histogram — Agent ReAct/FC 步数分布
  KB_CHUNK_COUNT     Gauge     — 知识库 Chunk 数量

面试亮点：
  - 三种指标类型对应不同观测维度：Counter=总量、Histogram=分布、Gauge=快照
  - p50/p95/p99 由 Histogram bucket 自动计算
  - prometheus_client 直接注册，而非黑盒 instrumentator
"""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

# ── 指标定义 ───────────────────────────────────────────

# 请求量
REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "RAG 系统请求总量",
    labelnames=["endpoint", "status"],
)

# 请求延迟（单位：秒）
# bucket 覆盖 100ms ~ 60s，p50/p95/p99 均在范围内
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG 系统请求延迟（秒）",
    labelnames=["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# 检索分数分布
RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "检索片段平均分分布",
    labelnames=[],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# LLM 调用延迟（单位：秒）
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "LLM 调用延迟（秒）",
    labelnames=["model", "api_type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# Agent 步数分布
AGENT_STEPS = Histogram(
    "rag_agent_steps",
    "Agent ReAct/FC 步数分布",
    labelnames=[],
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

# 知识库大小（Gauge，非累计值）
KB_CHUNK_COUNT = Gauge(
    "rag_kb_chunk_count",
    "知识库 Chunk 总数",
    labelnames=[],
)

# Agent Token 消耗（Histogram，每会话一次）
AGENT_TOKEN_USAGE = Histogram(
    "rag_agent_token_usage",
    "Agent 会话 Token 消耗",
    labelnames=[],
    buckets=[100, 500, 2000, 8000, 16384, 32768],
)

# Agent 终止原因分布（Counter）
AGENT_BUDGET_REASON = Counter(
    "rag_agent_budget_reason",
    "Agent 终止原因分布",
    labelnames=["reason"],
)


# ── 便捷函数 ───────────────────────────────────────────


def record_request(endpoint: str, status: int, duration: float) -> None:
    """记录一次请求的指标"""
    status_label = "ok" if 200 <= status < 400 else "error"
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status_label).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_retrieval(scores: list) -> None:
    """记录检索分数分布"""
    if scores:
        avg_score = sum(scores) / len(scores)
        RETRIEVAL_SCORE.observe(avg_score)


def record_llm_latency(duration: float, model: str = "unknown", api_type: str = "openai") -> None:
    """记录 LLM 调用延迟"""
    LLM_LATENCY.labels(model=model, api_type=api_type).observe(duration)


def record_agent_steps(steps: int) -> None:
    """记录 Agent 步数"""
    AGENT_STEPS.observe(steps)


def record_kb_size(count: int) -> None:
    """设置知识库大小（启动时调用一次）"""
    KB_CHUNK_COUNT.set(count)


def record_agent_token_usage(tokens: int) -> None:
    """记录 Agent 会话 Token 消耗"""
    AGENT_TOKEN_USAGE.observe(tokens)


def record_agent_budget_reason(reason: str) -> None:
    """记录 Agent 终止原因"""
    AGENT_BUDGET_REASON.labels(reason=reason).inc()


def get_metrics_registry() -> CollectorRegistry:
    """返回 Prometheus 注册表，供 /metrics 端点使用"""
    return REGISTRY


F = TypeVar("F", bound=Callable[..., Any])


def record_latency(histogram: Histogram, labels: dict[str, str] | None = None) -> Callable[[F], F]:
    """装饰器：记录函数耗时到指定 Histogram

    用法:
        @record_latency(LLM_LATENCY, labels={"model": "deepseek"})
        def call_llm():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.monotonic() - start
                if labels:
                    histogram.labels(**labels).observe(elapsed)
                else:
                    histogram.observe(elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator

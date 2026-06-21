"""
可观测性模块 — OpenTelemetry 追踪 + Prometheus 指标 + structlog 结构化日志

三大支柱：
  1. Tracing:   OpenTelemetry ConsoleSpanExporter，手动 span 装饰器
  2. Metrics:   Prometheus Counter/Histogram/Gauge
  3. Logging:   structlog processor pipeline → JSON

面试价值：
  - "可观测性"是微服务和 AI 系统的高频面试考点
  - 手动 span 而非 auto-instrumentation，展示对分布式追踪的理解
  - structlog 的 processor pipeline 是 Chain-of-Responsibility 模式的典型应用
  - Prometheus 三种指标类型 (Counter/Histogram/Gauge) 对应不同观测维度
"""

from .logging_config import bind_trace_id, setup_structlog
from .metrics import (
    AGENT_STEPS,
    KB_CHUNK_COUNT,
    LLM_LATENCY,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    RETRIEVAL_SCORE,
    get_metrics_registry,
    record_agent_steps,
    record_llm_latency,
    record_request,
    record_retrieval,
)
from .tracing import TraceContextManager, get_current_trace_id, init_tracing, trace_call

__all__ = [
    # tracing
    "trace_call",
    "TraceContextManager",
    "get_current_trace_id",
    "init_tracing",
    # metrics
    "REQUESTS_TOTAL",
    "REQUEST_LATENCY",
    "RETRIEVAL_SCORE",
    "LLM_LATENCY",
    "AGENT_STEPS",
    "KB_CHUNK_COUNT",
    "record_request",
    "record_retrieval",
    "record_llm_latency",
    "record_agent_steps",
    "get_metrics_registry",
    # logging
    "setup_structlog",
    "bind_trace_id",
]

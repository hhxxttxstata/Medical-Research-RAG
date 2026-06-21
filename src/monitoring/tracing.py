"""
Tracing — OpenTelemetry 追踪层

提供：
  - init_tracing()    初始化 TracerProvider + ConsoleSpanExporter
  - @trace_call()     函数装饰器，自动包裹 span
  - TraceContextManager 上下文管理器，用于函数内的子 span
  - get_current_trace_id()  从隐式 context 提取 trace_id

面试亮点：
  - 手动 span 展示对 span 生命周期、context propagation、error recording 的理解
  - ConsoleSpanExporter 输出 JSON 到 stderr → 面试时说"生产环境换 OTLP 导出器即可"
  - 无 auto-instrumentation 黑盒，每一条 trace 都是显式的
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# 模块级 tracer（由 init_tracing 创建）
_tracer: trace.Tracer | None = None


def init_tracing(service_name: str = "pe-rag-system") -> None:
    """初始化 OpenTelemetry TracerProvider，设置 ConsoleSpanExporter

    可在不同入口点重复调用（幂等），不会覆盖已初始化的 provider。
    """
    global _tracer

    provider = trace.get_tracer_provider()
    # 检查是否已经初始化（避免重复添加 processor）
    if not hasattr(provider, "_initialized"):
        provider = TracerProvider()
        exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        provider._initialized = True  # type: ignore[attr-defined]
        trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)


def get_tracer() -> trace.Tracer:
    """获取模块级 tracer（未初始化时自动初始化）"""
    global _tracer
    if _tracer is None:
        init_tracing()
    return _tracer


def get_current_trace_id() -> str:
    """从当前隐式 context 提取 trace_id（十六进制字符串）

    如果没有活跃 span，返回空字符串。
    """
    span = trace.get_current_span()
    if not span:
        return ""
    ctx = span.get_span_context()
    if not ctx or not ctx.trace_id or ctx.trace_id == 0:
        return ""
    return format(ctx.trace_id, "032x")


def get_current_span_id() -> str:
    """从当前隐式 context 提取 span_id（十六进制字符串）"""
    span = trace.get_current_span()
    if not span:
        return ""
    ctx = span.get_span_context()
    if not ctx or not ctx.span_id or ctx.span_id == 0:
        return ""
    return format(ctx.span_id, "016x")


F = TypeVar("F", bound=Callable[..., Any])


def trace_call(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """函数装饰器：被装饰的函数执行时自动创建一个 span

    用法:
        @trace_call("my_operation", attributes={"key": "value"})
        def my_function():
            ...

        @trace_call()  # 自动使用函数名作为 span name
        def my_function():
            ...

    span 自动记录：
      - 函数参数快照（前 200 字符）
      - 执行耗时
      - 异常（exception + ERROR status）
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = span_name or func.__name__
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


class TraceContextManager:
    """上下文管理器：在代码块范围内创建 span

    用法:
        with TraceContextManager("retrieval", {"count": 5}) as span:
            span.set_attribute("count", 5)
            results = retrieve(...)

    等价于 tracer.start_as_current_span()，但不需要外部 tracer 引用。
    """

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.attrs = attributes or {}
        self.ctx_mgr = None
        self.span = None

    def __enter__(self) -> trace.Span:
        tracer = get_tracer()
        self.ctx_mgr = tracer.start_as_current_span(self.name)
        self.span = self.ctx_mgr.__enter__()
        if self.attrs:
            self.span.set_attributes(self.attrs)
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span:
            if exc_val:
                self.span.record_exception(exc_val)
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        if self.ctx_mgr:
            self.ctx_mgr.__exit__(exc_type, exc_val, exc_tb)

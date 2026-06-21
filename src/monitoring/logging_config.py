"""
Logging Config — structlog 结构化日志配置

提供：
  - setup_structlog()  配置全局 structlog processor pipeline
  - bind_trace_id()    将当前 trace_id 写入 structlog context

processor pipeline（Chain-of-Responsibility 模式）：
  1. merge_contextvars  — 注入 context 变量（trace_id）
  2. add_log_level      — 添加日志级别
  3. TimeStamper        — ISO 8601 时间戳
  4. UnicodeDecoder     — 确保非 ASCII 字符正确编码
  5. JSONRenderer       — 最终输出 JSON 行

面试亮点：
  - structlog 的 processor pipeline 是 Chain-of-Responsibility 模式
  - 比 loguru 的 monkey-patching 架构更清晰，面试更容易解释
  - 与 OpenTelemetry context 天然集成，trace_id 自动注入
"""

import os

import structlog

from .tracing import get_current_trace_id


def _add_trace_context(logger: structlog.BoundLogger, method_name: str, event_dict: dict) -> dict:
    """structlog processor：自动注入当前 trace_id 和 span_id

    从 OpenTelemetry 隐式 context 读取，无需显式传递。
    """
    trace_id = get_current_trace_id()
    if trace_id:
        event_dict["trace_id"] = trace_id

    from .tracing import get_current_span_id

    span_id = get_current_span_id()
    if span_id:
        event_dict["span_id"] = span_id

    return event_dict


def setup_structlog(
    service_name: str = "pe-rag-system",
    log_level: str = "INFO",
    force_json: bool = False,
) -> None:
    """配置全局 structlog processor pipeline

    Args:
        service_name: 服务名称，会被加入每个日志事件
        log_level: 默认日志级别
        force_json: 强制使用 JSON 渲染器（即使不是生产模式）
    """
    is_dev = not force_json and os.getenv("STRUCTLOG_FORMAT", "json").lower() == "console"

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        # 注入 trace_id / span_id
        _add_trace_context,
    ]

    # 开发模式：带颜色的控制台输出；生产模式：JSON
    if is_dev:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 写入一条启动日志
    log = structlog.get_logger()
    log.info("structlog initialized", service=service_name, format="json" if not is_dev else "console")


def bind_trace_id() -> None:
    """将当前 trace_id 绑定到 structlog context

    在请求处理开始时调用一次，后续同一请求的所有 structlog 调用
    都会自动携带 trace_id。
    """
    trace_id = get_current_trace_id()
    if trace_id:
        structlog.contextvars.bind_contextvars(trace_id=trace_id)


def get_logger() -> structlog.stdlib.BoundLogger:
    """获取 structlog logger 实例"""
    return structlog.get_logger()

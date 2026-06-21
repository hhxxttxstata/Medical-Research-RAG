"""
工具包
提供可供 Agent 调用的各种工具
"""

from .diagnosis_tool import DiagnosisTool
from .report_generator import ReportGenerator

__all__ = ["ReportGenerator", "DiagnosisTool"]

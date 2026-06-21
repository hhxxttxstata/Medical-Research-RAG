"""
MCP Server — 将 RAG 系统的工具以 MCP (Model Context Protocol) 形式暴露

通过 FastMCP 将 Agent 中注册的工具（DiagnosisTool、RAGPipeline 等）
发布为 MCP Server，支持 stdio 和 SSE 两种传输协议。

MCP Host（如 Claude Desktop、VS Code、自研 MCP 客户端）可：
  1. 调用 tools/list 发现可用工具
  2. 调用 tools/call 执行具体工具

启动方式：
  # stdio 模式（默认，适合 Claude Desktop 配置）
  python -m src.mcp_server

  # SSE 模式（适合 HTTP 远程调用）
  python -m src.mcp_server --transport sse --port 8123

Claude Desktop 配置示例 (claude_desktop_config.json):
  {
    "mcpServers": {
      "pe-rag-system": {
        "command": "python",
        "args": ["-m", "src.mcp_server"],
        "env": { "PYTHONPATH": "d:/Pulmonary_embolism_system" }
      }
    }
  }

面试价值：
  - MCP 是最新的 AI Agent 工具开放协议（Anthropic 提出，2024年底发布）
  - 展示对 MCP 的理解：Tool Discovery + Tool Calling + Transport 层分离
  - 与现有 Agent 系统的 Tool 基类天然兼容，一行属性即可接入
  - Claude Desktop、Cursor、VS Code 等主流工具可直接调用
"""

import argparse
import atexit
import json
import os
import sys
from typing import Any

# ── 确保项目根目录在 Python 路径中 ────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 可观测性 ────────────────────────────────────────
from opentelemetry import trace

from .monitoring.logging_config import setup_structlog
from .monitoring.tracing import init_tracing

# ── MCP Server 工厂 ──────────────────────────────────


def create_mcp_server(
    server_name: str = "pe-rag-system",
    data_dir: str = "",
    persist_dir: str = "",
    chunk_min_chars: int = 300,
    chunk_max_chars: int = 500,
    top_k: int = 5,
) -> Any:
    """创建 FastMCP 实例，注册项目中的所有工具

    Returns:
        FastMCP 实例，调用 .run(transport=...) 启动服务
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name=server_name,
        instructions="""# 肺栓塞 RAG 系统 — MCP 工具集

提供以下工具供 MCP Host 调用：

1. **retrieve_knowledge** — 从肺栓塞知识库中检索相关文档片段
2. **rag_query** — 完整的 RAG 问答（检索 + 生成）
3. **diagnose_pulmonary_embolism** — 肺栓塞 AI 影像诊断
4. **generate_report** — 生成结构化报告（部署/排查/会议）

## 使用场景
- 检索肺栓塞相关的医学知识
- 对 CTPA 影像进行 AI 辅助诊断
- 生成项目相关的结构化文档
""",
        debug=False,
    )

    # ── 初始化 RAG Pipeline（共享实例） ──

    from src.rag_pipeline import RAGPipeline

    resolved_data_dir = os.path.abspath(data_dir or os.path.join(_PROJECT_ROOT, "data"))
    resolved_persist_dir = os.path.abspath(persist_dir or os.path.join(_PROJECT_ROOT, "chroma_db"))

    pipeline = RAGPipeline(
        data_dir=resolved_data_dir,
        persist_dir=resolved_persist_dir,
        chunk_min_chars=chunk_min_chars,
        chunk_max_chars=chunk_max_chars,
        top_k=top_k,
    )

    # 自动初始化知识库（如不为空则跳过）
    try:
        count = pipeline.initialize_knowledge_base(force_reindex=False)
        _log(f"📚 知识库就绪: {count} 个 Chunk")
    except Exception as e:
        _log(f"⚠️  知识库初始化跳过: {e}")

    # ── 工具 1: retrieve_knowledge ──────────────────────

    @mcp.tool(
        name="retrieve_knowledge",
        description="从肺栓塞知识库检索与查询最相关的文档片段（Top-K），返回原文、来源和相似度分数",
    )
    def retrieve_knowledge(
        query: str,
        top_k: int = 5,
    ) -> str:
        """检索知识库

        Args:
            query: 搜索查询（中文或英文）
            top_k: 返回的片段数量（1-20，默认 5）

        Returns:
            JSON 字符串，包含检索到的文档片段列表
        """
        try:
            chunks = pipeline.retriever.retrieve(query, top_k=max(1, min(top_k, 20)))
            if not chunks:
                return json.dumps({"success": True, "count": 0, "results": []}, ensure_ascii=False)

            results = []
            for c in chunks:
                results.append(
                    {
                        "id": c["id"],
                        "text": c["text"],
                        "score": round(c["score"], 4),
                        "source": c["metadata"].get("filename", "未知"),
                        "page": c["metadata"].get("page", ""),
                    }
                )
            return json.dumps(
                {"success": True, "count": len(results), "results": results}, ensure_ascii=False, indent=2
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    # ── 工具 2: rag_query ─────────────────────────────

    @mcp.tool(
        name="rag_query",
        description="完整的 RAG 问答：先检索知识库，再基于检索结果生成回答，返回回答文本、来源和检索质量指标",
    )
    def rag_query(
        question: str,
        top_k: int = 5,
    ) -> str:
        """RAG 问答

        Args:
            question: 用户问题
            top_k: 检索的文档片段数量

        Returns:
            JSON 字符串，包含回答、引用来源、检索质量指标
        """
        try:
            result = pipeline.query(question, top_k=max(1, min(top_k, 20)))
            sources = []
            for s in result.get("sources", []):
                sources.append(
                    {
                        "id": s["id"],
                        "filename": s["metadata"].get("filename", "未知"),
                        "score": round(s["score"], 4),
                        "text_preview": s["text"][:300],
                    }
                )

            return json.dumps(
                {
                    "success": not result.get("error"),
                    "question": result["question"],
                    "answer": result["answer"],
                    "is_refusal": result.get("is_refusal", False),
                    "error": result.get("error"),
                    "sources": sources,
                    "metrics": {
                        "retrieval_time": round(result.get("time", 0), 2),
                        "num_sources": len(sources),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    # ── 工具 3: diagnose_pulmonary_embolism ──────────

    def _run_diagnosis(file_path: str, return_mask: bool = True) -> str:
        """执行肺栓塞诊断（封装为同步函数供 MCP 调用）"""
        from src.tools.diagnosis_tool import DiagnosisTool

        tool = DiagnosisTool()
        result = tool.run(file_path=file_path, return_mask=return_mask)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @mcp.tool(
        name="diagnose_pulmonary_embolism",
        description="对 CTPA 影像（NIfTI 格式 .nii/.nii.gz）进行 AI 肺栓塞诊断，返回概率评分、风险等级和分割信息",
    )
    def diagnose_pulmonary_embolism(
        file_path: str,
        return_mask: bool = True,
    ) -> str:
        """肺栓塞 AI 影像诊断

        Args:
            file_path: NIfTI 格式的 CTPA 影像文件完整路径（.nii 或 .nii.gz）
            return_mask: 是否返回分割掩膜数据（默认 true）

        Returns:
            JSON 字符串，包含诊断结果、概率、风险等级
        """
        return _run_diagnosis(file_path, return_mask)

    # ── 工具 4: generate_report ──────────────────────

    @mcp.tool(
        name="generate_report",
        description="根据检索结果生成结构化报告（部署报告 / 问题排查报告 / 会议纪要），返回 Markdown 格式文本",
    )
    def generate_report(
        report_type: str,
        content: str,
        topic: str = "",
    ) -> str:
        """生成结构化报告

        Args:
            report_type: 报告类型 —— "deployment"（部署）/ "troubleshoot"（排查）/ "meeting"（会议）
            content: 资料内容（检索结果文本或用户提供的材料）
            topic: 报告主题（可选，留空自动推断）

        Returns:
            JSON 字符串，包含 report 字段（Markdown 格式）
        """
        from src.tools.report_generator import ReportGenerator

        tool = ReportGenerator()
        if pipeline and pipeline.generator:
            tool.set_generator(pipeline.generator)
        result = tool.run(report_type=report_type, content=content, topic=topic)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    return mcp


# ── CLI 启动入口 ─────────────────────────────────────


def main():
    # 初始化可观测性
    init_tracing(service_name="pe-rag-system-mcp")
    setup_structlog(service_name="pe-rag-system-mcp", force_json=False)
    # MCP 退出时确保 span 被刷出
    atexit.register(lambda: trace.get_tracer_provider().force_flush())

    parser = argparse.ArgumentParser(description="肺栓塞 RAG 系统 — MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输协议（默认 stdio，适合 Claude Desktop；sse 适合 HTTP 远程调用）",
    )
    parser.add_argument("--port", type=int, default=8123, help="SSE 模式端口号（默认 8123）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="SSE 模式监听地址（默认 127.0.0.1）")
    parser.add_argument("--data-dir", type=str, default="", help="知识库文档目录（默认 data/）")
    parser.add_argument("--persist-dir", type=str, default="", help="向量数据库持久化目录（默认 chroma_db/）")
    parser.add_argument("--top-k", type=int, default=5, help="默认检索数量（默认 5）")

    args = parser.parse_args()

    _log("=" * 50)
    _log("  🚀 肺栓塞 RAG 系统 — MCP Server")
    _log(f"  📡 传输协议: {args.transport}")
    _log("=" * 50)

    mcp = create_mcp_server(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        top_k=args.top_k,
    )

    if args.transport == "sse":
        _log(f"  🌐 启动 SSE 服务: http://{args.host}:{args.port}/sse")
        _log(f"  📖 消息端点: http://{args.host}:{args.port}/messages/")
        mcp.run(transport="sse", mount_path=None)
    else:
        _log("  🔌 启动 stdio 模式...")
        mcp.run(transport="stdio")


def _log(msg: str) -> None:
    """打印日志（MCP stdio 模式下，stderr 不会被 MCP Host 拦截）"""
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

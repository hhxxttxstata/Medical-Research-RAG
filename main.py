"""
RAG 系统主入口
支持命令行交互式问答和知识库初始化
"""

import argparse
import os
import sys
from pathlib import Path

# Windows GBK 编码兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from src.agent import Agent
from src.logger import get_logger

# ── 可观测性 ────────────────────────────────────────
from src.monitoring.logging_config import setup_structlog
from src.monitoring.tracing import init_tracing
from src.rag_pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG 检索增强生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 初始化知识库（默认 chunk 500-800 字）
  python main.py --init

  # 使用不同 chunk size 重建索引
  python main.py --rebuild --chunk-size 300
  python main.py --rebuild --chunk-size 800

  # 单次查询（指定 top-k）
  python main.py --query "肺结节的CT影像特征是什么？"
  python main.py --query "U-Net的核心创新是什么？" --top-k 8

  # 交互式问答
  python main.py --interactive

  # Agent 模式（自动判断意图，支持报告生成等工具调用）
  python main.py --agent --query "生成一份RAG系统的部署报告"
  python main.py --agent --query "排查部署失败的原因" --report-type troubleshoot
  python main.py --agent -i  (交互式 Agent 模式)

  # 查看系统状态
  python main.py --status

  # 查看运行统计
  python main.py --stats
        """,
    )

    parser.add_argument("--init", action="store_true", help="初始化知识库")
    parser.add_argument("--rebuild", action="store_true", help="重建索引（清空后重新导入）")
    parser.add_argument("--query", type=str, help="输入查询问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="启动交互式问答模式")
    parser.add_argument("--status", action="store_true", help="查看系统状态")
    parser.add_argument("--stats", action="store_true", help="查看运行统计")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回的片段数量（默认: 3，评估推荐值）")
    parser.add_argument("--agent", action="store_true", help="启用 Agent 模式（自动判断意图，支持报告生成等工具调用）")
    parser.add_argument(
        "--report-type",
        type=str,
        choices=["deployment", "troubleshoot", "meeting"],
        default=None,
        help="强制指定报告类型（deployment:部署报告, troubleshoot:问题排查报告, meeting:会议纪要）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk 最大字数（默认: 500，实际范围为 300-500 字）",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Embedding 提供者（默认: local）",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding 模型名称（默认: BAAI/bge-small-zh-v1.5）",
    )

    # ── MCP Server 参数 ──
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="启动 MCP Server（Model Context Protocol），将工具暴露给 MCP Host（如 Claude Desktop）",
    )
    parser.add_argument(
        "--mcp-transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP 传输协议（默认 stdio，适合 Claude Desktop；sse 适合 HTTP 远程调用）",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8123,
        help="MCP SSE 模式端口号（默认 8123）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化可观测性（MCP 模式在 mcp_main() 中自行初始化）
    if not args.mcp:
        init_tracing(service_name="pe-rag-system-cli")
        setup_structlog(service_name="pe-rag-system-cli", force_json=False)

    # ── MCP Server 模式：直接启动，不创建本地 pipeline ──
    if args.mcp:
        from src.mcp_server import main as mcp_main

        # 替换 argv 让 mcp_server 解析自己的参数
        sys.argv = [
            "mcp_server.py",
            "--transport",
            args.mcp_transport,
            "--port",
            str(args.mcp_port),
            "--data-dir",
            "data",
            "--persist-dir",
            "chroma_db",
            "--top-k",
            str(args.top_k),
        ]
        mcp_main()
        return

    # 计算 chunk 范围
    chunk_min = max(200, args.chunk_size - 200)
    chunk_max = args.chunk_size

    # 初始化管道
    pipeline = RAGPipeline(
        data_dir="data",
        persist_dir="chroma_db",
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        chunk_min_chars=chunk_min,
        chunk_max_chars=chunk_max,
    )

    # 查看系统状态
    if args.status:
        show_status(pipeline)
        return

    # 查看运行统计
    if args.stats:
        logger = get_logger()
        logger.print_summary()
        return

    # 初始化知识库
    if args.init:
        pipeline.initialize_knowledge_base(force_reindex=False)
        return

    # 重建索引
    if args.rebuild:
        pipeline.initialize_knowledge_base(force_reindex=True)
        return

    # 单次查询（Agent 模式）
    if args.agent and args.query:
        agent_main(pipeline, args.query, args.report_type)
        return

    # 单次查询
    if args.query:
        result = pipeline.query(args.query)
        pipeline.print_result(result)
        return

    # 交互式模式（Agent 模式）
    if args.agent:
        interactive_agent_mode(pipeline)
        return

    # 交互式模式
    if args.interactive:
        interactive_mode(pipeline)
        return

    # 默认：检查状态并提示
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，请先运行: python main.py --init")
        print("   或者: python main.py --rebuild")
        print("   指定 chunk size: python main.py --init --chunk-size 300")
        print("   交互模式: python main.py --interactive\n")
    else:
        print(f"\n📊 知识库状态: {count} 个 Chunk 已就绪")
        print("💡 使用方式:")
        print('   python main.py --query "你的问题"')
        print("   python main.py --interactive")
        print("   python main.py --stats  (查看运行统计)")
        print('   python main.py --agent --query "生成部署报告"  (Agent 模式)')
        print("   python main.py --help   (查看完整帮助)")


def show_status(pipeline: RAGPipeline):
    """显示系统状态"""
    print("\n" + "=" * 60)
    print("  📊 RAG 系统状态")
    print("=" * 60)

    data_dir = pipeline.data_dir
    doc_files = (
        list(Path(data_dir).glob("*.md")) + list(Path(data_dir).glob("*.txt")) + list(Path(data_dir).glob("*.pdf"))
    )

    print(f"\n📁 数据目录: {data_dir}")
    print(f"📄 文档数量: {len(doc_files)}")
    print(f"   - Markdown: {len(list(Path(data_dir).glob('*.md')))}")
    print(f"   - TXT: {len(list(Path(data_dir).glob('*.txt')))}")
    print(f"   - PDF: {len(list(Path(data_dir).glob('*.pdf')))}")

    chunk_count = pipeline.vector_store.count()
    print(f"\n🧩 Chunk 数量: {chunk_count}")
    print(f"📐 Chunk 范围: {pipeline.chunk_min_chars}-{pipeline.chunk_max_chars} 字")

    print(f"\n🔧 Embedding: {pipeline.embedding_provider.__class__.__name__}")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print(f"💾 数据库: {pipeline.persist_dir}")
    print("📝 日志目录: logs/")
    print("=" * 60 + "\n")


def agent_main(pipeline: RAGPipeline, query: str, report_type: str = None):
    """
    Agent 模式入口：自动判断意图 -> 检索上下文 -> 调用工具 -> 输出结果
    """
    # 确保知识库已初始化
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，正在初始化...")
        pipeline.initialize_knowledge_base()
        count = pipeline.vector_store.count()
        if count == 0:
            print("❌ 知识库初始化失败，请检查 data/ 目录")
            return

    # 初始化 Agent
    agent = Agent(rag_pipeline=pipeline)

    # 如果用户强制指定了报告类型，直接走对应报告生成流程
    if report_type:
        print("\n" + "=" * 60)
        print("  🔧 Agent 模式（指定报告类型）")
        print("=" * 60)
        print(f"  📋 强制报告类型: {report_type}\n")

        # 检索上下文
        print("📡 正在检索知识库，获取相关上下文...")
        retrieved = pipeline.retriever.retrieve(query, top_k=8)
        print(f"  ✅ 检索到 {len(retrieved)} 个相关片段\n")

        # 拼接内容
        content_parts = []
        sources_summary_parts = []
        for i, chunk in enumerate(retrieved, 1):
            meta = chunk["metadata"]
            filename = meta.get("filename", "未知")
            page = meta.get("page", "")
            source_label = f"[{i}] {filename}"
            if page:
                source_label += f" (第{page}页)"
            if chunk.get("score", 0) > 0.2:
                sources_summary_parts.append(source_label)
                content_parts.append(
                    f"### 来源 [{i}]: {filename}" + (f" | 页码: {page}" if page else "") + f"\n{chunk['text']}\n"
                )

        content = "\n\n".join(content_parts)
        sources_summary = ", ".join(sources_summary_parts) if sources_summary_parts else "知识库检索"

        report_type_name = {"deployment": "部署", "troubleshoot": "问题排查", "meeting": "会议"}[report_type]

        # 调用报告生成工具
        print(f"📝 正在生成{report_type_name}报告...\n")
        generator = agent.tools["generate_report"]
        generator.set_generator(pipeline.generator)
        result = generator.run(
            report_type=report_type,
            content=content,
            topic=query,
            context={
                "sources_summary": sources_summary,
                "source_count": len(retrieved),
            },
        )

        if result.get("success"):
            print(result["report"])
            print(f"\n{'=' * 60}")
            print(f"  ✅ {report_type_name}报告生成完成，基于 {len(retrieved)} 个检索片段")
            print(f"{'=' * 60}\n")
        return

    # 自动意图识别流程
    result = agent.process(query, session_id="cli_main")

    if result.get("agent_handled"):
        tool_result = result.get("result", {})
        if tool_result.get("success"):
            print(tool_result["report"])
            print(f"\n{'=' * 60}")
            report_types = {
                "deployment": "部署报告",
                "troubleshoot": "问题排查报告",
                "meeting": "会议纪要",
            }
            rtype = result.get("report_type", "")
            print(
                f"  ✅ {report_types.get(rtype, '报告')}生成完成，基于 {len(result.get('retrieved_chunks', []))} 个检索片段"
            )
            print(f"{'=' * 60}\n")
        else:
            print(f"\n❌ 工具调用失败: {tool_result}")
    else:
        # 未匹配到工具 -> 走常规 RAG 流程
        print("\n🔄 未匹配到专用工具，转为常规 RAG 问答\n")
        rag_result = pipeline.query(query)
        pipeline.print_result(rag_result)


def interactive_agent_mode(pipeline: RAGPipeline):
    """交互式 Agent 模式"""
    print("\n" + "=" * 60)
    print("  🤖 RAG Agent 交互模式")
    print("  🧠 自动识别意图 -> 检索知识库 -> 调用工具 -> 生成报告")
    print("=" * 60)

    # 确保知识库已初始化
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，正在初始化...")
        pipeline.initialize_knowledge_base()
        count = pipeline.vector_store.count()
        if count == 0:
            print("❌ 知识库初始化失败，请检查 data/ 目录")
            return

    agent = Agent(rag_pipeline=pipeline)

    print(f"\n📚 知识库已就绪: {count} 个 Chunk")
    print("💡 输入 'quit' 退出，'reload' 重载")
    print("💡 支持以下场景:")
    print("   - 部署/上线/发布 -> 自动生成部署报告")
    print("   - 问题/故障/报错 -> 自动生成问题排查报告")
    print("   - 会议/讨论 -> 自动生成会议纪要")
    print("   - 其他问题 -> 常规 RAG 问答")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if question.lower() == "reload":
                pipeline.initialize_knowledge_base(force_reindex=True)
                continue

            result = agent.process(question, session_id="cli_interactive")

            if result.get("agent_handled"):
                tool_result = result.get("result", {})
                if tool_result.get("success"):
                    print(tool_result["report"])
                    print(f"\n{'=' * 60}")
                    report_types = {
                        "deployment": "部署报告",
                        "troubleshoot": "问题排查报告",
                        "meeting": "会议纪要",
                    }
                    rtype = result.get("report_type", "")
                    print(f"  ✅ {report_types.get(rtype, '报告')}生成完成")
                    print(f"{'=' * 60}\n")
                else:
                    print(f"\n❌ 工具调用失败: {tool_result}")
            else:
                print("\n🔄 转为常规 RAG 问答\n")
                rag_result = pipeline.query(question)
                pipeline.print_result(rag_result)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def interactive_mode(pipeline: RAGPipeline):
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("  🤖 RAG 交互式问答系统")
    print("=" * 60)

    # 检查知识库状态
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，正在初始化...")
        pipeline.initialize_knowledge_base()
        count = pipeline.vector_store.count()
        if count == 0:
            print("❌ 知识库初始化失败，请检查 data/ 目录")
            return

    print(f"\n📚 知识库已就绪: {count} 个 Chunk")
    print(f"📐 Chunk: {pipeline.chunk_min_chars}-{pipeline.chunk_max_chars} 字")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print("💡 输入 'quit' 退出，'reload' 重载，'stats' 查看统计")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if question.lower() == "reload":
                pipeline.initialize_knowledge_base(force_reindex=True)
                continue
            if question.lower() == "stats":
                logger = get_logger()
                logger.print_summary()
                continue

            result = pipeline.query(question)
            pipeline.print_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()

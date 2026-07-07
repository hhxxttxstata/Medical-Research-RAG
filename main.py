"""
RAG 系统主入口 — CLI 交互式问答和知识库管理
"""

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import get_logger
from src.rag_pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG 检索增强生成系统",
        epilog="""
使用示例:
  python main.py --init                       # 初始化知识库
  python main.py --rebuild --chunk-size 300    # 重建索引
  python main.py --query "肺结节的CT影像特征？"  # 单次查询
  python main.py -i                            # 交互式问答
  python main.py --status                      # 系统状态
        """,
    )
    parser.add_argument("--init", action="store_true", help="初始化知识库")
    parser.add_argument("--rebuild", action="store_true", help="重建索引")
    parser.add_argument("--query", type=str, help="输入查询问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式问答")
    parser.add_argument("--status", action="store_true", help="系统状态")
    parser.add_argument("--stats", action="store_true", help="运行统计")
    parser.add_argument("--top-k", type=int, default=5, help="检索数量")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk 最大字数")
    parser.add_argument("--embedding-provider", type=str, choices=["local", "openai"], default="local")
    parser.add_argument("--embedding-model", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    chunk_min = max(200, args.chunk_size - 200)
    chunk_max = args.chunk_size

    pipeline = RAGPipeline(
        data_dir="data",
        persist_dir="chroma_db",
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        chunk_min_chars=chunk_min,
        chunk_max_chars=chunk_max,
    )

    if args.status:
        show_status(pipeline)
        return
    if args.stats:
        get_logger().print_summary()
        return
    if args.init:
        pipeline.initialize_knowledge_base(force_reindex=False)
        return
    if args.rebuild:
        pipeline.initialize_knowledge_base(force_reindex=True)
        return
    if args.query:
        result = pipeline.query(args.query)
        pipeline.print_result(result)
        return
    if args.interactive:
        interactive_mode(pipeline)
        return

    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，运行: python main.py --init\n")
    else:
        print(f"\n📊 知识库: {count} 个 Chunk")
        print('💡 python main.py --query "你的问题"')
        print("   python main.py -i\n")


def show_status(pipeline: RAGPipeline):
    print("\n" + "=" * 60)
    print("  📊 RAG 系统状态")
    print("=" * 60)
    print(f"\n📄 文档数: {len(list(Path(pipeline.data_dir).glob('*.md')) + list(Path(pipeline.data_dir).glob('*.txt')) + list(Path(pipeline.data_dir).glob('*.pdf')))}")
    print(f"🧩 Chunk: {pipeline.vector_store.count()}")
    print(f"🔧 Embedding: {pipeline.embedding_provider.__class__.__name__}")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print("=" * 60 + "\n")


def interactive_mode(pipeline: RAGPipeline):
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n⚠️  知识库为空，正在初始化...")
        pipeline.initialize_knowledge_base()
        count = pipeline.vector_store.count()

    print(f"\n📚 知识库: {count} 个 Chunk")
    print("输入 'quit' 退出\n")

    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            result = pipeline.query(question)
            pipeline.print_result(result)
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()

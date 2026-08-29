"""
Agentic RAG Demo CLI — 直接调用 LangGraph Agentic RAG（无需启动 Web 服务）

用法:
    python scripts/agent_demo.py "肺栓塞CTPA的直接征象有哪些？"   # 单次查询
    python scripts/agent_demo.py                                 # 交互模式
    python scripts/agent_demo.py "问题" --json                   # 输出原始 JSON

依赖: 本地 milvus_db/ 索引 + .env 中的 DEEPSEEK_API_KEY（无 key/无网时自动降级规则 fallback）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("MILVUS_LITE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def build_service():
    from src.agent_service import AgentQueryService
    from src.rag_pipeline import RAGPipeline

    print("⏳ 初始化知识库连接（Milvus Lite）...")
    pipeline = RAGPipeline(
        data_dir="data",
        top_k=8,
        enable_rewrite=False,
        enable_reranker=False,
        milvus_lite=True,
        vector_backend="milvus",
    )
    count = pipeline.vector_store.count()
    print(f"📚 知识库: {count} chunks")
    pipeline.retriever._ensure_bm25_index()
    print("⏳ 加载 Agentic（reranker + agent，首次约 10-20s）...")
    svc = AgentQueryService(pipeline)
    return svc


def show(result: dict, pretty: bool = True) -> None:
    if not pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("\n" + "=" * 64)
    badge = "✅ ACCEPT" if result["status"] == "ACCEPT" else "🚫 ABSTAIN"
    print(f"  {badge}   route: {' → '.join(result['route'])}")
    print(
        f"  iterations={result['iterations']}  grader_called={result['grader_called']}  "
        f"latency={result['latency_ms']}ms"
    )
    if result.get("abstain_reason"):
        print(f"  拒答原因: {result['abstain_reason']}")
    print("-" * 64)
    print(result["answer"])
    print("-" * 64)
    print(f"📚 final_evidence ({len(result['evidence'])})")
    for i, e in enumerate(result["evidence"], 1):
        print(f"  {i}. [{e['score']:.3f}] {e['filename']}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Agentic RAG Demo CLI")
    ap.add_argument("question", nargs="?", help="查询问题（缺省进入交互模式）")
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = ap.parse_args()

    svc = build_service()

    if args.question:
        show(svc.query(args.question), pretty=not args.json)
        return

    print("\n💬 Agentic RAG 交互模式（输入 exit 退出）\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            break
        try:
            show(svc.query(q), pretty=not args.json)
        except Exception as e:
            print(f"  ❌ 查询失败: {e}")


if __name__ == "__main__":
    main()

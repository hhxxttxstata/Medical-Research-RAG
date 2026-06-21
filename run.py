"""
快速启动脚本
可直接运行:  python run.py
或配置环境变量:  API_HOST=0.0.0.0  API_PORT=8000  API_RELOAD=true  python run.py
"""

import os
import sys

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import uvicorn

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print("=" * 60)
    print("  🚀 RAG 知识库问答系统")
    print("  📚 基于检索增强生成 + Agent 工具调度")
    print("=" * 60)
    print(f"\n🌐 服务地址: http://{host}:{port}")
    print(f"📖 API 文档: http://{host}:{port}/docs")
    print(f"📋 Redoc:    http://{host}:{port}/redoc")
    print("\n💡 启动参数:")
    print(f"   API_HOST={host}")
    print(f"   API_PORT={port}")
    print(f"   API_RELOAD={'开启' if reload else '关闭'}")
    print()

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=["src", "data", "."] if reload else None,
    )

"""
rebuild_index.py — 全量重建知识库索引（与后端 startup 同参数）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/rebuild_index.py

纪律: 串行独占（Milvus Lite 单进程锁）——运行前停掉后端/其他评测进程。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MILVUS_LITE", "true")


def main():
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        data_dir="data",
        top_k=8,
        chunk_min_chars=300,
        chunk_max_chars=500,
        retriever_mode="hybrid",
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
        milvus_lite=True,
    )
    t0 = time.time()
    count = pipeline.initialize_knowledge_base(force_reindex=True)
    print(f"\n✅ 重建完成: {count} chunks, 耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    # 释放连接（Milvus Lite 单进程独占纪律）
    pipeline.close()
    print("✅ 连接已释放", flush=True)


if __name__ == "__main__":
    main()

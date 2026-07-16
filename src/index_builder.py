"""
索引构建器

统一编排双索引的构建/重建流程：
  1. 加载文档 → 切分 Chunk
  2. Lucene BM25（Whoosh 磁盘索引）
  3. Embedding → Milvus 向量数据库（或 ChromaDB）

面试价值：
  - 展示双索引设计的架构理解：倒排索引 + 向量索引的统一编排
  - IndexBuilder 负责索引全生命周期，与检索器解耦
  - 增量更新：新增文档时只追加，避免全量重建
"""

import os
import time

from .document_loader import load_and_process_from_dir, load_documents_from_dir
from .document_processor import process_document
from .embeddings import get_embedding_provider
from .lucene_bm25 import LuceneBM25Index
from .milvus_store import MilvusStore
from .text_splitter import split_document


class IndexBuilder:
    """索引构建器

    全量构建:
        builder = IndexBuilder(data_dir="data", vector_backend="milvus", bm25_index_dir="lucene_bm25_index")
        builder.build_all()

    增量添加:
        builder.add_document("data/new_paper.pdf")

    仅重建 BM25:
        builder.build_bm25_only(chunks)
    """

    def __init__(
        self,
        data_dir: str = "data",
        vector_backend: str = "milvus",
        bm25_index_dir: str = "lucene_bm25_index",
        collection_name: str = "rag_docs",
        embedding_provider: str = "local",
        embedding_model: str | None = None,
        chunk_min_chars: int = 300,
        chunk_max_chars: int = 500,
        use_smart_chunking: bool = True,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_lite: bool = False,
        dim: int = 768,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.bm25_index_dir = bm25_index_dir
        self.bm25_index_dir = bm25_index_dir
        self.collection_name = collection_name
        self.chunk_min_chars = chunk_min_chars
        self.chunk_max_chars = chunk_max_chars
        self.use_smart_chunking = use_smart_chunking
        self.dim = dim

        self.embedding_provider = get_embedding_provider(embedding_provider, embedding_model)
        self.lucene_bm25 = LuceneBM25Index(index_dir=bm25_index_dir)

        self.vector_store = MilvusStore(
            collection_name=collection_name,
            dim=dim,
            host=milvus_host,
            port=milvus_port,
            use_lite=milvus_lite,
        )

    # ── 全量构建 ────────────────────────────────────

    def build_all(self) -> int:
        """全量重建双索引

        Returns:
            构建的 Chunk 总数
        """
        print("\n" + "=" * 60)
        print("  🔨 开始构建双索引 (BM25 + Vector)")
        print("=" * 60 + "\n")

        start = time.time()

        # 1. 加载 & 切分文档
        print("📖 步骤 1/3: 加载文档...")
        if self.use_smart_chunking:
            processed_docs = load_and_process_from_dir(self.data_dir)
            if not processed_docs:
                print("  ❌ 未找到支持的文档")
                return 0
            all_chunks = []
            for pdoc in processed_docs:
                all_chunks.extend(pdoc.get("small_chunks", []))
            print(f"  ✅ 共处理 {len(processed_docs)} 个文档 → {len(all_chunks)} 个 Chunk\n")
        else:
            documents = load_documents_from_dir(self.data_dir)
            if not documents:
                print("  ❌ 未找到支持的文档")
                return 0
            all_chunks = []
            for doc in documents:
                chunks = split_document(
                    doc,
                    chunk_min_chars=self.chunk_min_chars,
                    chunk_max_chars=self.chunk_max_chars,
                )
                all_chunks.extend(chunks)
            print(f"  ✅ 共加载 {len(documents)} 个文档 → {len(all_chunks)} 个 Chunk\n")

        if not all_chunks:
            return 0

        # 2. 构建 Lucene BM25 索引
        print("🔤 步骤 2/3: 构建 Lucene BM25 索引...")
        self.lucene_bm25.rebuild(all_chunks)
        bm25_count = self.lucene_bm25.get_total_docs()
        print(f"  ✅ BM25 索引完成: {bm25_count} 个文档\n")

        # 3. 生成 Embedding → 写入向量存储
        print("🧬 步骤 3/3: 生成 Embedding & 写入向量存储...")
        texts = [chunk["text"] for chunk in all_chunks]
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_provider.embed(batch_texts, prefix="passage: ")
            all_embeddings.extend(batch_embeddings)
            print(f"  📊 Embedding: {min(i + batch_size, len(texts))}/{len(texts)}")

        # 先清空向量集合
        self.vector_store.delete_collection()

        # 写入新数据
        self.vector_store.add_chunks(all_chunks, all_embeddings)
        vector_count = self.vector_store.count()
        print(f"  ✅ 向量索引完成: {vector_count} 条\n")

        elapsed = time.time() - start
        print("=" * 60)
        print(f"  🎉 双索引构建完成! 耗时 {elapsed:.1f}s")
        print(f"     BM25: {bm25_count} 篇 | 向量: {vector_count} 条")
        print("=" * 60 + "\n")

        return len(all_chunks)

    # ── 单索引构建 ───────────────────────────────────

    def build_bm25_only(self, chunks: list[dict] | None = None, clear_first: bool = True):
        """仅重建 BM25 索引

        Args:
            chunks: 待索引的 Chunk 列表（None = 从向量存储获取）
            clear_first: 是否清空后重建
        """
        if chunks is None:
            chunks = self.vector_store.get_all_documents()
        if not chunks:
            print("  ⚠️ 无数据，跳过 BM25 构建")
            return

        if clear_first:
            self.lucene_bm25.rebuild(chunks)
        else:
            self.lucene_bm25.index_chunks(chunks)
        print(f"  ✅ BM25 索引: {self.lucene_bm25.get_total_docs()} 篇文档")

    def build_vector_only(self, chunks: list[dict] | None = None, clear_first: bool = True):
        """仅重建向量索引

        Args:
            chunks: 待写入的 Chunk 列表（None = 自动加载）
            clear_first: 是否清空后重建
        """
        if chunks is None:
            # 从文档重新加载
            if self.use_smart_chunking:
                processed_docs = load_and_process_from_dir(self.data_dir)
                chunks = []
                for pdoc in processed_docs:
                    chunks.extend(pdoc.get("small_chunks", []))
            else:
                documents = load_documents_from_dir(self.data_dir)
                chunks = []
                for doc in documents:
                    c = split_document(
                        doc,
                        chunk_min_chars=self.chunk_min_chars,
                        chunk_max_chars=self.chunk_max_chars,
                    )
                    chunks.extend(c)

        if not chunks:
            print("  ⚠️ 无数据，跳过向量索引构建")
            return

        if clear_first:
            self.vector_store.delete_collection()

        texts = [chunk["text"] for chunk in chunks]
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_provider.embed(batch_texts, prefix="passage: ")
            embeddings.extend(batch_embeddings)

        self.vector_store.add_chunks(chunks, embeddings)
        print(f"  ✅ 向量索引: {self.vector_store.count()} 条")

    # ── 增量添加 ────────────────────────────────────

    def add_document(self, file_path: str) -> int:
        """向双索引增量添加一个文档

        Args:
            file_path: 文档路径（PDF/MD/TXT）

        Returns:
            新增的 Chunk 数
        """
        if not os.path.isfile(file_path):
            print(f"  ❌ 文件不存在: {file_path}")
            return 0

        # 1. 处理文档
        if self.use_smart_chunking:
            pdoc = process_document(file_path)
            chunks = pdoc.get("small_chunks", [])
        else:
            from .document_loader import load_document

            doc = load_document(file_path)
            chunks = split_document(
                doc,
                chunk_min_chars=self.chunk_min_chars,
                chunk_max_chars=self.chunk_max_chars,
            )

        if not chunks:
            print(f"  ⚠️ 文档切分为空: {file_path}")
            return 0

        # 2. 写入 BM25
        self.lucene_bm25.index_chunks(chunks)

        # 3. 生成 Embedding 并写入向量存储
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_provider.embed(texts, prefix="passage: ")
        self.vector_store.add_chunks(chunks, embeddings)

        count = len(chunks)
        print(f"  ✅ 已增量添加文档: {os.path.basename(file_path)} ({count} 个 Chunk)")
        return count

    # ── 清理 ────────────────────────────────────────

    def close(self):
        """释放所有资源"""
        self.lucene_bm25.close()
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

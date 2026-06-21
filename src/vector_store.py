"""
向量数据库模块
使用 Chroma 存储 Chunk 向量和元数据
"""

import os
from typing import Any

import chromadb
from chromadb.config import Settings


class VectorStore:
    """向量数据库封装"""

    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "rag_docs"):
        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def get_or_create_collection(self):
        """获取或创建集合"""
        client = self._get_client()
        try:
            self._collection = client.get_collection(self.collection_name)
            count = self._collection.count()
            print(f"  📂 使用已有集合 '{self.collection_name}' (现存 {count} 条)")
        except Exception:
            self._collection = client.create_collection(self.collection_name)
            print(f"  🆕 创建新集合 '{self.collection_name}'")
        return self._collection

    def delete_collection(self):
        """删除集合"""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
            print(f"  🗑️ 已删除集合 '{self.collection_name}'")
        except Exception as e:
            print(f"  ⚠️ 删除集合失败: {e}")

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """添加 Chunk 到向量数据库"""
        collection = self.get_or_create_collection()
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            texts = [c["text"] for c in batch_chunks]
            metadatas = [c["metadata"] for c in batch_chunks]

            collection.add(
                embeddings=batch_embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """检索最相似的 Chunk"""
        collection = self.get_or_create_collection()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunks.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i],  # 余弦相似度
                    }
                )
        return chunks

    def count(self) -> int:
        """获取集合中的 Chunk 数量"""
        try:
            collection = self.get_or_create_collection()
            return collection.count()
        except Exception:
            return 0

    def close(self):
        """释放 ChromaDB 客户端资源，解除文件锁"""
        self._collection = None
        if self._client is not None:
            try:
                # ChromaDB PersistentClient 没有显式 close()，
                # 清空引用让 GC 回收底层 SQLite/DuckDB 连接
                self._client = None
            except Exception:
                pass

    def get_all_documents(self) -> list[dict[str, Any]]:
        """获取集合中所有文档（不含 embedding，用于 BM25 建索引）"""
        try:
            collection = self.get_or_create_collection()
            # 不传 where 参数 = 获取全部
            results = collection.get(include=["documents", "metadatas"])
            chunks = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    chunks.append(
                        {
                            "id": results["ids"][i],
                            "text": results["documents"][i] if results["documents"] else "",
                            "metadata": results["metadatas"][i] if results["metadatas"] else {},
                        }
                    )
            return chunks
        except Exception:
            return []

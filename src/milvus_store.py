"""
Milvus 向量数据库封装

支持两种连接模式：
  1. Milvus Standalone（Docker 部署）— 生产模式，连接远程服务
  2. Milvus Lite（pip install pymilvus[milvus-lite]）— 嵌入式模式，文件持久化

集合设计:
  - 名称: rag_docs（可自定义）
  - 字段:
      - pk (INT64, 自动 ID)
      - chunk_id (VARCHAR, 唯一标识)
      - embedding (FLOAT_VECTOR, dim=768)
      - text (VARCHAR, max_length=65535)
      - metadata (JSON)
  - 索引: IVF_FLAT (nlist=128) 或 HNSW (M=16, efConstruction=200)
  - 度量: IP（内积，与 L2 归一化的 cosine 等价）

面试亮点：
  - 从 ChromaDB（嵌入式）到 Milvus（分布式）的架构演进思路
  - 索引类型选择：IVF_FLAT 适合 10 万级中等规模，HNSW 适合更高精度要求
  - Collection 管理 + 标量过滤 + 一致性级别
"""

import json
import os
import time
from typing import Any

# ── 集合与索引常量 ─────────────────────────────────

_DEFAULT_COLLECTION = "rag_docs"
_DEFAULT_DIM = 768
_MAX_RETRIES = 3
_RETRY_DELAY = 2


class MilvusStore:
    """Milvus 向量数据库封装

    使用方法:
        store = MilvusStore()
        store.add_chunks(chunks, embeddings)
        results = store.similarity_search(query_embedding, top_k=5)
        store.close()
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        dim: int = _DEFAULT_DIM,
        host: str = "localhost",
        port: str = "19530",
        use_lite: bool = False,
        lite_db_path: str = "milvus_db",
    ):
        """
        Args:
            collection_name: Milvus 集合名称
            dim: 向量维度（需与实际模型输出一致）
            host: Milvus 服务地址（Standalone 模式）
            port: Milvus 服务端口（Standalone 模式）
            use_lite: True = 使用 Milvus Lite（嵌入式，无需 Docker）
            lite_db_path: Milvus Lite 数据持久化目录
        """
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.use_lite = use_lite
        self.lite_db_path = os.path.abspath(lite_db_path) if use_lite else ""

        self._connected = False
        self._collection = None

    # ── 连接管理 ────────────────────────────────────

    def _connect(self):
        """连接 Milvus 服务"""
        if self._connected:
            return
        self._ensure_imports()
        self._do_connect()
        self._ensure_collection()

    def _ensure_imports(self):
        """延迟导入 pymilvus 模块，避免没有安装时报 ImportError"""
        global MilvusClient, DataType, IndexType, MetricType
        if self.use_lite:
            try:
                from pymilvus import DataType, IndexType, MetricType, MilvusClient

                self._client_class = MilvusClient
                self._use_lite = True
            except ImportError:
                print("  ⚠️ pymilvus[milvus-lite] 未安装，回退到 Milvus 服务模式")
                from pymilvus import DataType, IndexType, MetricType, MilvusClient

                self._client_class = MilvusClient
                self._use_lite = False
        else:
            from pymilvus import DataType, IndexType, MetricType, MilvusClient

            self._client_class = MilvusClient
            self._use_lite = False

        self.DataType = DataType
        self.IndexType = IndexType
        self.MetricType = MetricType

    def _do_connect(self):
        """建立与 Milvus 的连接"""
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                if self._use_lite:
                    os.makedirs(self.lite_db_path, exist_ok=True)
                    db_file = os.path.join(self.lite_db_path, "milvus.db")
                    self._client = self._client_class(uri=db_file)
                else:
                    uri = f"http://{self.host}:{self.port}"
                    self._client = self._client_class(uri=uri)
                self._connected = True
                print(f"  📦 Milvus {'Lite' if self._use_lite else 'Standalone'} 连接成功")
                return
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
        print(f"  ⚠️ Milvus 连接失败: {last_error}. 将使用 ChromaDB 回退。")
        self._connected = False

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        if not self._connected:
            return
        try:
            # 检查集合是否存在
            collections = self._client.list_collections()
            if self.collection_name in collections:
                self._collection = self.collection_name
                cnt = self._client.get_collection_stats(self.collection_name).get("row_count", 0)
                print(f"  📂 打开 Milvus 集合 '{self.collection_name}' (现存 {cnt} 条)")
                return
        except Exception:
            pass

        # 创建集合
        try:
            schema = self._client.create_schema(
                auto_id=True,
                enable_dynamic_field=False,
            )
            schema.add_field("pk", self.DataType.INT64, is_primary=True)
            schema.add_field("chunk_id", self.DataType.VARCHAR, max_length=256)
            schema.add_field("embedding", self.DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("text", self.DataType.VARCHAR, max_length=65535)
            schema.add_field("metadata", self.DataType.JSON)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=self.IndexType.IVF_FLAT,
                metric_type=self.MetricType.IP,
                params={"nlist": 128},
            )

            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self._collection = self.collection_name
            print(f"  🆕 创建 Milvus 集合 '{self.collection_name}' (dim={self.dim})")
        except Exception as e:
            print(f"  ⚠️ 创建 Milvus 集合失败: {e}")
            self._collection = None

    # ── 数据写入 ────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ):
        """批量写入 Chunk 向量

        Args:
            chunks: [{"chunk_id": str, "text": str, "metadata": dict}, ...]
            embeddings: 与 chunks 等长的向量列表
        """
        self._connect()
        if not self._connected or self._collection is None:
            print("  ⚠️ Milvus 不可用，跳过向量写入")
            return

        batch_size = 100
        inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            data = []
            for j, c in enumerate(batch_chunks):
                meta = c.get("metadata", {})
                data.append(
                    {
                        "chunk_id": str(c.get("chunk_id", c.get("id", f"chunk_{i + j}"))),
                        "embedding": batch_embeddings[j],
                        "text": c.get("text", ""),
                        "metadata": json.loads(json.dumps(meta, ensure_ascii=False)),
                    }
                )

            try:
                self._client.insert(self.collection_name, data)
                inserted += len(data)
            except Exception as e:
                print(f"  ⚠️ Milvus 批量插入失败 (batch {i}): {e}")

        if inserted > 0:
            # 写入后刷新索引
            try:
                self._client.flush(self.collection_name)
            except Exception:
                pass
            print(f"  ✅ 插入 {inserted} 条向量到 Milvus 集合 '{self.collection_name}'")

    # ── 向量检索 ────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """检索最相似的 Chunk

        Args:
            query_embedding: 查询向量
            top_k: 返回 top-K 结果
            where: Milvus 标量过滤表达式（如 {"filename": "doc.pdf"}）

        Returns:
            [{"id": str, "text": str, "metadata": dict, "score": float}, ...]
        """
        self._connect()
        if not self._connected or self._collection is None:
            return []

        try:
            search_params = {
                "metric_type": self.MetricType.IP,
                "params": {"nprobe": 16},
            }

            results = self._client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                search_params=search_params,
                limit=top_k,
                output_fields=["chunk_id", "text", "metadata"],
                filter=where or None,
            )

            chunks = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    meta = entity.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                    chunks.append(
                        {
                            "id": entity.get("chunk_id", str(hit.get("id", ""))),
                            "text": entity.get("text", ""),
                            "metadata": meta,
                            "score": float(hit.get("distance", 0)),
                        }
                    )
            return chunks

        except Exception as e:
            print(f"  ⚠️ Milvus 检索失败: {e}")
            return []

    # ── 管理方法 ────────────────────────────────────

    def count(self) -> int:
        """获取集合中的向量数量"""
        self._connect()
        if not self._connected or self._collection is None:
            return 0
        try:
            stats = self._client.get_collection_stats(self.collection_name)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def delete_collection(self, name: str | None = None):
        """删除集合"""
        target = name or self.collection_name
        self._connect()
        if not self._connected:
            return
        try:
            self._client.drop_collection(target)
            if target == self.collection_name:
                self._collection = None
            print(f"  🗑️ 已删除 Milvus 集合 '{target}'")
        except Exception as e:
            print(f"  ⚠️ 删除 Milvus 集合失败: {e}")

    def get_all_documents(self) -> list[dict[str, Any]]:
        """获取集合中所有文档（不含 embedding，用于 BM25 建索引）"""
        self._connect()
        if not self._connected or self._collection is None:
            return []

        try:
            results = self._client.query(
                collection_name=self.collection_name,
                output_fields=["chunk_id", "text", "metadata"],
                limit=10000,
            )
            chunks = []
            for r in results:
                meta = r.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        meta = {}
                chunks.append(
                    {
                        "id": r.get("chunk_id", ""),
                        "text": r.get("text", ""),
                        "metadata": meta,
                    }
                )
            return chunks
        except Exception as e:
            print(f"  ⚠️ Milvus 全量查询失败: {e}")
            return []

    def get_collection(self, name: str):
        """按名称检查集合是否存在（兼容接口）"""
        self._connect()
        if not self._connected:
            return None
        try:
            collections = self._client.list_collections()
            if name in collections:
                return name
        except Exception:
            pass
        return None

    def list_collections(self) -> list:
        """列出所有集合"""
        self._connect()
        if not self._connected:
            return []
        try:
            return self._client.list_collections()
        except Exception:
            return []

    def close(self):
        """释放资源"""
        if self._connected and hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False
        self._collection = None

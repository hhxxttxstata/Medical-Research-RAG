"""
knowledge_base.py — 知识库管理

多集合、标签、版本管理。
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _normalize_collection(vs_collection) -> dict:
    """将后端返回的集合对象统一为 dict 格式

    兼容 ChromaDB（返回 Chroma Collection 对象）和 Milvus（返回字符串或 None）。
    """
    if vs_collection is None:
        return {"name": "", "metadata": {}, "count": lambda: 0}
    if isinstance(vs_collection, str):
        # Milvus 返回集合名称字符串 → 暂时无权获取元数据
        return {"name": vs_collection, "metadata": {}, "count": lambda: 0}
    # ChromaDB Collection 对象
    return {
        "name": getattr(vs_collection, "name", ""),
        "metadata": getattr(vs_collection, "metadata", {}) or {},
        "count": lambda: getattr(vs_collection, "count", lambda: 0)(),
    }


class KnowledgeBase:
    """知识库管理

    通过 VectorStore/MilvusStore 的操作实现集合/标签/版本管理，
    不绑定特定后端。
    """

    # ── 集合管理 ──────────────────────────────────────

    @staticmethod
    def list_collections(vector_store) -> list[dict]:
        """列出所有集合及其元数据"""
        try:
            collections = vector_store.list_collections()
            result = []
            for col in collections:
                col_info = _normalize_collection(col)
                meta = col_info["metadata"]
                result.append(
                    {
                        "name": col_info["name"],
                        "chunk_count": col_info["count"](),
                        "version": meta.get("version", 1),
                        "tags": meta.get("tags", []),
                        "created_at": meta.get("created_at", ""),
                        "updated_at": meta.get("updated_at", ""),
                    }
                )
            return result
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []

    @staticmethod
    def get_collection_info(vector_store, name: str) -> dict | None:
        """获取单个集合的详细信息"""
        try:
            col = vector_store.get_collection(name)
            if col is None:
                return None
            col_info = _normalize_collection(col)
            meta = col_info["metadata"]
            return {
                "name": col_info["name"],
                "chunk_count": col_info["count"](),
                "version": meta.get("version", 1),
                "tags": meta.get("tags", []),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "embedding_model": meta.get("embedding_model", ""),
                "dimension": meta.get("dimension", 0),
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None

    @staticmethod
    def create_collection(
        vector_store,
        name: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """创建新集合

        Args:
            name: 集合名称
            tags: 文档标签列表
            metadata: 额外元数据（如 embedding_model, dimension）

        Returns:
            集合信息 dict
        """
        meta = {
            "version": 1,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        try:
            col = vector_store.create_collection(name, metadata=meta)
            col_info = _normalize_collection(col)
            logger.info(f"📚 创建集合: {name} (tags={tags})")
            return {
                "name": col_info["name"],
                "chunk_count": 0,
                "version": 1,
                "tags": tags or [],
                "created_at": meta["created_at"],
            }
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    @staticmethod
    def delete_collection(vector_store, name: str) -> bool:
        """删除集合"""
        try:
            vector_store.delete_collection(name)
            logger.info(f"🗑️ 删除集合: {name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    # ── 版本管理 ──────────────────────────────────────

    @staticmethod
    def bump_version(vector_store, collection_name: str) -> int:
        """递增集合版本号（在 rebuild 后调用）

        Returns:
            新版本号
        """
        from datetime import datetime

        try:
            col = vector_store.get_collection(collection_name)
            if col is None:
                return 1
            col_info = _normalize_collection(col)
            meta = dict(col_info["metadata"])
            version = meta.get("version", 0) + 1
            meta["version"] = version
            meta["updated_at"] = datetime.now().isoformat()
            vector_store.update_collection_metadata(collection_name, meta)
            logger.info(f"📌 集合 {collection_name} 版本: {version}")
            return version
        except Exception as e:
            logger.error(f"版本更新失败: {e}")
            return -1

    @staticmethod
    def get_version(vector_store, collection_name: str) -> int:
        """获取集合当前版本号"""
        info = KnowledgeBase.get_collection_info(vector_store, collection_name)
        return info.get("version", 1) if info else 0

    # ── 标签管理 ──────────────────────────────────────

    @staticmethod
    def get_all_tags(vector_store) -> list[str]:
        """获取所有集合中使用的标签"""
        tags = set()
        collections = KnowledgeBase.list_collections(vector_store)
        for col in collections:
            tags.update(col.get("tags", []))
        return sorted(tags)

    @staticmethod
    def filter_by_tags(
        vector_store,
        query_embedding: list[float],
        tags: list[str],
        top_k: int = 10,
    ) -> list[dict]:
        """按标签过滤 + 向量搜索

        使用 ChromaDB 的 where 过滤功能。
        """
        filter_dict = {"tags": {"$in": tags}} if tags else None
        try:
            return vector_store.similarity_search(query_embedding, top_k=top_k, where=filter_dict)
        except Exception as e:
            logger.error(f"标签过滤检索失败: {e}")
            return []

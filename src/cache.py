"""
cache.py — 多级缓存系统

设计：
  Redis 优先，内存 LRU fallback，连接失败时透明降级。
  嵌入缓存、检索缓存、回答缓存三层独立 TTL。
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  Redis 客户端（单例）
# ═══════════════════════════════════════════════════


class RedisClient:
    """Redis 客户端单例，连接失败时返回 None（不抛异常）"""

    _client = None
    _enabled = False

    @classmethod
    def get_client(cls):
        if cls._client is not None:
            return cls._client if cls._enabled else None

        url = os.getenv("REDIS_URL", "")
        if not url:
            logger.info("Redis 未配置（REDIS_URL 为空），使用内存缓存")
            cls._enabled = False
            cls._client = None
            return None

        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            cls._client = client
            cls._enabled = True
            logger.info(f"Redis 连接成功: {url}")
            return client
        except Exception as e:
            logger.warning(f"Redis 连接失败，降级到内存缓存: {e}")
            cls._enabled = False
            cls._client = None
            return None

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled


# ═══════════════════════════════════════════════════
#  内存 LRU 缓存（线程安全，TTL 支持）
# ═══════════════════════════════════════════════════


class LRUCache:
    """带 TTL 的线程安全 LRU 缓存"""

    def __init__(self, maxsize: int = 1000, default_ttl: int | None = 3600):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = None

    def _get_lock(self):
        if self._lock is None:
            import threading

            self._lock = threading.Lock()
        return self._lock

    def get(self, key: str) -> Any | None:
        with self._get_lock():
            if key not in self._cache:
                return None
            value, expires = self._cache[key]
            if expires and time.time() > expires:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._get_lock():
            expires: float = 0
            ttl = ttl if ttl is not None else self._default_ttl
            if ttl:
                expires = time.time() + ttl
            self._cache[key] = (value, expires)
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._get_lock():
            self._cache.clear()

    def __len__(self) -> int:
        with self._get_lock():
            return len(self._cache)


# ═══════════════════════════════════════════════════
#  Key 工具
# ═══════════════════════════════════════════════════


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════
#  嵌入缓存
# ═══════════════════════════════════════════════════


class EmbeddingCache:
    """缓存文本 → embedding 向量的结果

    Redis key:  emb:{sha256(text)}
    内存 LRU:   最大 2000 条，TTL 24h
    """

    REDIS_PREFIX = "emb:"
    MEM_TTL = 86400  # 24h

    def __init__(self, maxsize: int = 2000):
        self._mem = LRUCache(maxsize=maxsize, default_ttl=self.MEM_TTL)

    def get(self, text: str) -> list[float] | None:
        key = _sha256(text)
        # Redis 优先
        if RedisClient.is_enabled():
            try:
                client = RedisClient.get_client()
                raw = client.get(self.REDIS_PREFIX + key)
                if raw:
                    import json

                    return json.loads(raw)
            except Exception:
                pass
        # 内存 fallback
        return self._mem.get(key)

    def set(self, text: str, embedding: list[float], ttl: int | None = None) -> None:
        key = _sha256(text)
        if RedisClient.is_enabled():
            try:
                import json

                client = RedisClient.get_client()
                client.setex(self.REDIS_PREFIX + key, ttl or self.MEM_TTL, json.dumps(embedding))
            except Exception:
                pass
        self._mem.set(key, embedding, ttl=ttl or self.MEM_TTL)


# ═══════════════════════════════════════════════════
#  检索结果缓存
# ═══════════════════════════════════════════════════


class RetrievalCache:
    """缓存 query → 检索结果

    命中后跳过整个检索阶段（向量+BM25+reranker）。
    精确匹配（相同 query + top_k）。

    Redis key:  ret:{sha256(query)}:{top_k}
    内存 LRU:   最大 500 条，TTL 1h
    """

    REDIS_PREFIX = "ret:"
    MEM_TTL = 3600  # 1h

    def __init__(self, maxsize: int = 500):
        self._mem = LRUCache(maxsize=maxsize, default_ttl=self.MEM_TTL)

    def get(self, query: str, top_k: int) -> list[dict] | None:
        key = f"{_sha256(query)}:{top_k}"
        if RedisClient.is_enabled():
            try:
                client = RedisClient.get_client()
                raw = client.get(self.REDIS_PREFIX + key)
                if raw:
                    import json

                    return json.loads(raw)
            except Exception:
                pass
        return self._mem.get(key)

    def set(self, query: str, top_k: int, chunks: list[dict], ttl: int | None = None) -> None:
        key = f"{_sha256(query)}:{top_k}"
        if RedisClient.is_enabled():
            try:
                import json

                client = RedisClient.get_client()
                client.setex(self.REDIS_PREFIX + key, ttl or self.MEM_TTL, json.dumps(chunks, default=str))
            except Exception:
                pass
        self._mem.set(key, chunks, ttl=ttl or self.MEM_TTL)

    def invalidate(self, query: str, top_k: int | None = None) -> None:
        """使缓存失效，在重建索引后调用"""
        prefix = _sha256(query)
        if RedisClient.is_enabled():
            try:
                client = RedisClient.get_client()
                if top_k:
                    client.delete(f"{self.REDIS_PREFIX}{prefix}:{top_k}")
                else:
                    for k in client.scan_iter(f"{self.REDIS_PREFIX}{prefix}:*"):
                        client.delete(k)
            except Exception:
                pass
        self._mem.clear()


# ═══════════════════════════════════════════════════
#  回答缓存（精确 + 语义匹配）
# ═══════════════════════════════════════════════════


class AnswerCache:
    """缓存 query → LLM 回答

    两层匹配：
      1. 精确匹配（相同 query）
      2. 语义匹配（>threshold 余弦相似度）

    Redis key (精确):  ans:{sha256(query)}
    Redis key (语义):  ans_sem:{sha256(query)} → embedding
    内存 LRU:          最大 1000 条，TTL 30min，兼顾可用性和新鲜度
    """

    REDIS_PREFIX = "ans:"
    MEM_TTL = 1800  # 30min
    SEMANTIC_THRESHOLD = 0.85

    def __init__(self, maxsize: int = 1000, embedding_fn=None):
        self._mem = LRUCache(maxsize=maxsize, default_ttl=self.MEM_TTL)
        # embedding_fn(query) -> list[float]
        self._embedding_fn = embedding_fn

    def get(self, query: str, threshold: float = SEMANTIC_THRESHOLD) -> Any | None:
        """精确匹配 + 语义相似度匹配"""
        exact_key = _sha256(query)

        # 1. 精确匹配
        result = self._get_exact(exact_key)
        if result is not None:
            return result

        # 2. 语义匹配（需要 embedding_fn）
        if not self._embedding_fn:
            return None

        try:
            q_emb = self._embedding_fn([query])[0]
        except Exception:
            return None

        cached = self._mem_list() if not RedisClient.is_enabled() else self._redis_list_all()
        for entry in cached:
            emb = entry.get("emb")
            if emb and self._cosine_sim(q_emb, emb) >= threshold:
                return entry.get("data")
        return None

    def set(self, query: str, data: Any, ttl: int | None = None) -> None:
        key = _sha256(query)
        ttl = ttl or self.MEM_TTL
        emb = self._get_emb(query)

        if RedisClient.is_enabled():
            try:
                import json

                client = RedisClient.get_client()
                client.setex(f"{self.REDIS_PREFIX}{key}", ttl, json.dumps(data, default=str))
                if emb:
                    client.setex(f"{self.REDIS_PREFIX}sem:{key}", ttl, json.dumps(emb))
            except Exception:
                pass

        self._mem.set(key, {"data": data, "emb": emb}, ttl=ttl)

    def _get_exact(self, key: str) -> Any | None:
        if RedisClient.is_enabled():
            try:
                import json

                client = RedisClient.get_client()
                raw = client.get(self.REDIS_PREFIX + key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        mem = self._mem.get(key)
        if mem:
            return mem.get("data")
        return None

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-10)

    def _get_emb(self, query: str) -> list[float] | None:
        if not self._embedding_fn:
            return None
        try:
            return self._embedding_fn([query])[0]
        except Exception:
            return None

    def _mem_list(self) -> list[dict]:
        """遍历内存缓存中所有条目"""
        results = []
        with self._mem._get_lock():
            for val, _ in self._mem._cache.values():
                if isinstance(val, dict) and "data" in val:
                    results.append(val)
        return results

    def _redis_list_all(self) -> list[dict]:
        try:
            client = RedisClient.get_client()
            keys = client.keys(f"{self.REDIS_PREFIX}sem:*")
            results = []
            for k in keys:
                import json

                raw = client.get(k)
                if raw:
                    emb = json.loads(raw)
                    data_key = self.REDIS_PREFIX + k[len(f"{self.REDIS_PREFIX}sem:") :]
                    data_raw = client.get(data_key)
                    if data_raw:
                        results.append({"emb": emb, "data": json.loads(data_raw)})
            return results
        except Exception:
            return []

    def clear(self) -> None:
        self._mem.clear()
        if RedisClient.is_enabled():
            try:
                client = RedisClient.get_client()
                for k in client.scan_iter(f"{self.REDIS_PREFIX}*"):
                    client.delete(k)
            except Exception:
                pass


# ═══════════════════════════════════════════════════
#  缓存管理器（统一入口）
# ═══════════════════════════════════════════════════


class CacheManager:
    """缓存管理器，在 RAGPipeline 中作为唯一入口"""

    def __init__(self, embedding_fn=None):
        self.embedding = EmbeddingCache()
        self.retrieval = RetrievalCache()
        self.answer = AnswerCache(embedding_fn=embedding_fn)

    def invalidate_all(self) -> None:
        """重建索引后清除所有缓存"""
        self.embedding._mem.clear()
        self.retrieval._mem.clear()
        # Redis 不做全量清理（避免生产影响），只清内存

    def clear_retrieval(self, query: str | None = None) -> None:
        """增量文档后使相关检索结果失效"""
        if query:
            self.retrieval.invalidate(query)
        else:
            self.retrieval._mem.clear()

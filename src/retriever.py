"""
检索模块
实现 Top-k 检索，返回最相关的文档片段
支持纯向量检索和混合检索（向量 + BM25 + Reranker）

三级召回流水线：
  1. Query Rewriting  — LLM 将用户口语化问题改写为检索友好查询
  2. Hybrid Search    — 向量检索 + BM25 → RRF 融合
  3. Reranker         — Cross-encoder 或 LLM 对候选结果进行相关性重排序
"""

import json
import re
import time
from typing import Any

from .embeddings import EmbeddingProvider
from .lucene_bm25 import LuceneBM25Index
from .monitoring.metrics import record_retrieval

# ── 可观测性 ────────────────────────────────────────
from .monitoring.tracing import get_tracer
from .vector_store import VectorStore

# ── Query Rewriting 提示词 ──────────────────────────

_REWRITE_SYSTEM_PROMPT = """\
你是一个搜索查询优化助手。
请将用户的原始问题改写为更适合向量检索的搜索查询。

## 核心原则
1. **领域判断**：如果原始问题明显不属于本知识库覆盖的领域（医学影像、肺栓塞、深度学习医学应用等），则**直接返回原始问题，不要改写**。
2. **保留核心实体**：对领域内问题，保留疾病名、检查方法、症状、解剖部位等关键信息
3. 去掉口语化表达，改为关键词风格
4. 如果问题有多种解读角度，可以生成最多 3 条不同角度的查询
5. 每行一条查询，不要编号，不要多余内容

## 示例 — 领域内问题（应该改写）
原始问题：帮我看看这个CT片子有没有问题
改写后的查询：
肺栓塞 CTPA 影像诊断
CT 肺动脉造影 表现 判读

原始问题：这个病的治疗方法有哪些
改写后的查询：
肺栓塞 治疗 方法
肺栓塞 抗凝 溶栓 方案

## 示例 — 领域外问题（不应该改写）
原始问题：2025年全球经济增长率是多少？
改写后的查询：
2025年全球经济增长率是多少？

原始问题：如何配置Kubernetes集群的RBAC权限？
改写后的查询：
如何配置Kubernetes集群的RBAC权限？

原始问题：Python中如何使用async/await进行异步编程？
改写后的查询：
Python中如何使用async/await进行异步编程？"""

_REWRITE_USER_PROMPT = """原始问题：{query}
改写后的查询："""

# ── Reranker 提示词 ─────────────────────────────────

_RERANK_SYSTEM_PROMPT = """\
你是一个检索结果相关性评估助手。
请判断以下每个片段与用户问题的相关程度。

评分标准：
- 10分：直接回答问题的核心内容
- 7-9分：高度相关，包含问题涉及的实体或论述
- 4-6分：部分相关，涉及相关主题但不直接
- 1-3分：弱相关，仅提及个别关键词
- 0分：完全不相关

只输出 JSON 格式的评分列表，不要其他内容。"""

_RERANK_USER_PROMPT = """用户问题：{query}

请对以下每个片段的相关性打分（0-10分）：

{chunks_text}

输出 JSON："""


class Retriever:
    """检索器（三级召回：Query Rewriting + 向量+BM25 + Reranker）

    三级流水线：
      0. Query Rewriting（可选）— LLM 改写查询
      1. Hybrid Search — 向量检索 + BM25 关键词 + RRF 融合
      2. Reranker（可选）— LLM 对候选重新评分

    每一级都可独立开关。无 LLM 时自动跳过第 0/2 级，行为等同于纯向量检索。
    """

    # ── 默认提示词 ──────────
    REWRITE_SYSTEM_PROMPT = _REWRITE_SYSTEM_PROMPT
    REWRITE_USER_PROMPT = _REWRITE_USER_PROMPT
    RERANK_SYSTEM_PROMPT = _RERANK_SYSTEM_PROMPT
    RERANK_USER_PROMPT = _RERANK_USER_PROMPT

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        top_k: int = 20,
        bm25_weight: float = 0.5,
        generator=None,
        enable_rewrite: bool = True,
        enable_reranker: bool = True,
        rerank_top_k: int = 50,
        reranker=None,
        bm25_backend: str = "memory",
        bm25_index_dir: str = "lucene_bm25_index",
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        # 无 generator 时自动禁用 rewrite 和 LLM reranker
        self.generator = generator
        self.enable_rewrite = enable_rewrite if generator else False
        self.enable_reranker = enable_reranker if generator else False
        self.rerank_top_k = rerank_top_k
        self._reranker = reranker  # CrossEncoderReranker 实例
        self._bm25 = None  # 内存 BM25Okapi 或 LuceneBM25Index
        self._bm25_backend = bm25_backend if generator else "memory"
        self._bm25_index_dir = bm25_index_dir
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._original_query: str = ""  # 用于区分改写 query 和原始 query
        self._was_rewritten: bool = False  # 标记 rewrite 是否产生了有意义的改写
        self._out_of_domain: bool = False  # 标记 LLM 是否判断为领域外

    # ══════════════════════════════════════════════════
    #  主入口
    # ══════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """三级召回流水线

        1. Query Rewriting（可选）
        2. Hybrid Search（向量 + BM25 -> RRF）
        3. Reranker（可选）
        """
        k = top_k or self.top_k

        # ── 无 LLM → 纯向量检索（兼容旧版） ──
        if self.generator is None:
            query_embedding = self.embedding_provider.embed([query], prefix="query: ")[0]
            results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=k)
            for r in results:
                r.pop("_rrf_score", None)
                r.pop("_retriever", None)
            return results

        # ── 追踪 span ──
        tracer = get_tracer()
        with tracer.start_as_current_span("retriever.retrieve") as span:
            span.set_attribute("top_k", k)
            span.set_attribute("mode", "hybrid")
            span.set_attribute("rewrite_enabled", self.enable_rewrite)
            span.set_attribute("rerank_enabled", self.enable_reranker)
            start = time.monotonic()

            # 记录原始 query，供后续区分改写 query
            self._original_query = query

            # ── 阶段 0: Query Rewriting ──
            search_queries = self._rewrite_query(query) if self._can_rewrite() else [query]
            span.set_attribute("search_query_count", len(search_queries))
            span.set_attribute("was_rewritten", self._was_rewritten)

            # ── 阶段 1: Hybrid Search（每条改写 query 独立检索，结果去重合并） ──
            all_results = []
            seen_ids: set = set()
            fetch_k = max(k * 2, 20)

            for sq in search_queries:
                results = self._hybrid_retrieve(sq, fetch_k=fetch_k)
                for r in results:
                    if r["id"] not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r["id"])

            # ── 阶段 2: Reranker ──
            candidates = all_results[: max(self.rerank_top_k, k)]
            if self._can_rerank() and len(candidates) > k:
                reranked = self._rerank(query, candidates, k)
            else:
                reranked = candidates[:k]

            # 清理内部字段
            for r in reranked:
                r.pop("_rrf_score", None)
                r.pop("_retriever", None)

            elapsed = time.monotonic() - start
            span.set_attribute("total_retrieved", len(all_results))
            span.set_attribute("final_count", len(reranked))
            span.set_attribute("duration_ms", round(elapsed * 1000, 1))

            # 记录指标
            scores = [r.get("score", 0) for r in reranked]
            record_retrieval(scores)

            return reranked

    # ══════════════════════════════════════════════════
    #  Query Rewriting
    # ══════════════════════════════════════════════════

    def _can_rewrite(self) -> bool:
        """检查是否满足 rewrite 条件"""
        return self.enable_rewrite and self.generator is not None

    def _rewrite_query(self, query: str) -> list[str]:
        """调用 LLM 将用户问题改写为检索友好查询

        返回 1-3 条搜索 query。LLM 失败时回退到 [原始 query]。
        """
        self._was_rewritten = False
        self._out_of_domain = False
        try:
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": self.REWRITE_USER_PROMPT.format(query=query)},
                ],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception:
            return [query]

        parsed = self._parse_rewrite_response(response)
        if parsed:
            # 检查是否真的有改写——如果返回和原始 query 完全一样，视为领域外
            stripped = query.strip()
            if len(parsed) == 1 and parsed[0].strip() == stripped:
                print("  ✏️  Query Rewriting: '系统判定为领域外问题，保持原样'")
                self._out_of_domain = True
                return [query]
            print(f"  ✏️  Query Rewriting: '{query[:40]}...' → {parsed}")
            self._was_rewritten = True
            return parsed

        return [query]

    @staticmethod
    def _parse_rewrite_response(response: str) -> list[str] | None:
        """从 LLM 输出中解析改写后的查询列表"""
        lines = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉可能的编号前缀
            line = re.sub(r"^\d+[.、\)\s]+", "", line).strip()
            # 去掉首尾引号
            line = line.strip("\"'")
            if len(line) >= 4:
                lines.append(line)

        return lines if lines else None

    # ══════════════════════════════════════════════════
    #  Hybrid Search（向量 + BM25 → RRF）
    # ══════════════════════════════════════════════════

    def _hybrid_retrieve(self, query: str, fetch_k: int) -> list[dict[str, Any]]:
        """纯混合检索（不包含 rewrite / rerank），可被多条改写 query 重复调用"""
        # 1. 向量检索（语义）
        query_embedding = self.embedding_provider.embed([query], prefix="query: ")[0]
        vector_results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=fetch_k)
        for r in vector_results:
            r["_retriever"] = "vector"
            r["_vector_score"] = r.get("score", 0.0)

        # 2. BM25 检索（关键词）— 不论是否改写，都走 BM25
        bm25_results = self._bm25_retrieve(query, top_k=fetch_k)

        # 3. RRF 融合
        if bm25_results:
            fused = self._rrf_fusion(vector_results, bm25_results, top_k=fetch_k)
            # 如果融合结果不足，用向量结果补全
            if len(fused) < fetch_k:
                existing_ids = {r["id"] for r in fused}
                for r in vector_results:
                    if r["id"] not in existing_ids:
                        fused.append(r)
                        existing_ids.add(r["id"])
                        if len(fused) >= fetch_k:
                            break
            return fused
        else:
            return vector_results

    # ══════════════════════════════════════════════════
    #  Reranker
    # ══════════════════════════════════════════════════

    def _can_rerank(self) -> bool:
        """检查是否满足 reranker 条件（cross-encoder 或 LLM）"""
        return self.enable_reranker and (self._reranker is not None or self.generator is not None)

    def _rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """对候选 chunks 进行相关性重排序

        优先使用 Cross-encoder reranker（高效），
        回退到 LLM-as-reranker（贵，但准确）。
        """
        # ── Cross-encoder 优先 ──
        if self._reranker is not None and self._reranker.model_ready:
            try:
                return self._reranker.rerank(query, chunks, top_k)
            except Exception:
                # cross-encoder 失败时回退 LLM
                pass

        # ── LLM-as-reranker（回退方案） ──
        if self.generator is None:
            return chunks[:top_k]

        # 构建带编号的片段列表文本
        chunk_lines = []
        for i, c in enumerate(chunks):
            text_preview = c["text"][:200].replace("\n", " ")
            chunk_lines.append(f"[{i}] (来源: {c['metadata'].get('filename', '未知')}) {text_preview}")

        chunks_text = "\n".join(chunk_lines)
        user_prompt = self.RERANK_USER_PROMPT.format(query=query, chunks_text=chunks_text)

        try:
            response = self.generator.chat(
                messages=[
                    {"role": "system", "content": self.RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception:
            return chunks[:top_k]

        scores = self._parse_rerank_response(response, len(chunks))
        if scores is None or len(scores) != len(chunks):
            return chunks[:top_k]

        for i, c in enumerate(chunks):
            c["_rerank_score"] = scores[i] if i < len(scores) else 0
            c["score"] = scores[i]

        reranked = sorted(chunks, key=lambda x: x["_rerank_score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _parse_rerank_response(response: str, expected_count: int) -> list[float] | None:
        """从 LLM 输出中解析相关性评分列表"""
        text = response.strip()
        # 去除 markdown 代码块标记
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 数组
            m = re.search(r"\[[\s\S]*\]", text)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        if not isinstance(data, list):
            return None

        scores = []
        for item in data:
            if isinstance(item, dict):
                score = item.get("relevance")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            elif isinstance(item, (int, float)):
                scores.append(float(item))
            else:
                scores.append(0.0)

        # 如果评分数量不足，用 0 填充
        while len(scores) < expected_count:
            scores.append(0.0)

        return scores[:expected_count]

    # ══════════════════════════════════════════════════
    #  BM25 索引与检索（内存 BM25Okapi 或磁盘 Whoosh）
    # ══════════════════════════════════════════════════

    def _ensure_bm25_index(self):
        """懒初始化 BM25 索引（只跑一次）"""
        if self._bm25 is not None:
            return

        if self._bm25_backend == "lucene":
            # 磁盘 Lucene BM25（Whoosh）— 零内存增长
            self._bm25 = LuceneBM25Index(index_dir=self._bm25_index_dir)
            self._bm25_ids = []
            self._bm25_docs = []
            return

        # 传统内存 BM25（rank_bm25）
        all_chunks = self.vector_store.get_all_documents()
        if not all_chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self._bm25_ids = [c["id"] for c in all_chunks]
        self._bm25_docs = [c["text"] for c in all_chunks]
        tokenized_corpus = [self._bm25_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _bm25_tokenize(text: str) -> list[str]:
        if not text:
            return []
        tokens = []
        parts = re.split(r"([一-鿿])", text)
        buffer = ""
        for part in parts:
            if re.match(r"^[一-鿿]$", part):
                if buffer.strip():
                    tokens.extend(buffer.strip().lower().split())
                    buffer = ""
                tokens.append(part)
            else:
                buffer += part
        if buffer.strip():
            tokens.extend(buffer.strip().lower().split())
        return tokens

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self._ensure_bm25_index()
        if self._bm25 is None:
            return []

        if self._bm25_backend == "lucene":
            # 磁盘 Whoosh BM25 检索
            return self._bm25.search(query, top_k=top_k)

        # 传统内存 BM25Okapi 检索
        tokenized_query = self._bm25_tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        scored = list(zip(self._bm25_ids, self._bm25_docs, scores))
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:top_k]
        results = []
        for idx, text, score in top:
            chunk = self._find_chunk_by_id(idx)
            results.append(
                {
                    "id": idx,
                    "text": text,
                    "metadata": chunk["metadata"] if chunk else {},
                    "score": float(score),
                    "_retriever": "bm25",
                }
            )
        return results

    def _find_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        try:
            all_chunks = self.vector_store.get_all_documents()
            for c in all_chunks:
                if c["id"] == chunk_id:
                    return c
        except Exception:
            pass
        return None

    def get_bm25_info(self) -> dict[str, Any]:
        """获取 BM25 索引状态信息"""
        self._ensure_bm25_index()
        if self._bm25_backend == "lucene" and self._bm25:
            return {
                "bm25_ready": True,
                "backend": "lucene",
                "num_docs": self._bm25.get_total_docs(),
                "index_dir": self._bm25_index_dir,
            }
        return {
            "bm25_ready": self._bm25 is not None,
            "backend": "memory",
            "num_docs": len(self._bm25_ids) if self._bm25_ids else 0,
        }

    @staticmethod
    def _rrf_fusion(
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        vector_rank = {r["id"]: i + 1 for i, r in enumerate(vector_results)}
        bm25_rank = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}
        all_ids = set(vector_rank.keys()) | set(bm25_rank.keys())
        rrf_scores = {}
        id_to_result = {}
        # 预构建向量分数映射（所有 retriever 都可能贡献 base，但向量分只来自 vector_results）
        vector_scores = {}
        for r in vector_results:
            vs = r.get("_vector_score")
            vector_scores[r["id"]] = vs if vs is not None else r.get("score", 0.0)
        for r in vector_results:
            id_to_result[r["id"]] = r
        for r in bm25_results:
            if r["id"] not in id_to_result:  # 不覆盖向量结果
                id_to_result[r["id"]] = r
        for cid in all_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k + bm25_rank[cid])
            rrf_scores[cid] = score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            base = id_to_result.get(cid, {})
            results.append(
                {
                    "id": cid,
                    "text": base.get("text", ""),
                    "metadata": base.get("metadata", {}),
                    "score": round(rrf_scores[cid], 4),
                    "_vector_score": round(vector_scores.get(cid, 0.0), 4),  # 从独立映射取值，不依赖 base 的 _retriever
                    "_rrf_score": rrf_scores[cid],
                    "_retriever": "hybrid",
                }
            )
        return results

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
from typing import Any

from .embeddings import EmbeddingProvider
from .lucene_bm25 import LuceneBM25Index

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

## 示例 — 英文领域内问题（应该改写）
原始问题：What are the CT signs of pulmonary embolism?
改写后的查询：
CT signs pulmonary embolism
CT pulmonary angiography PE findings
pulmonary embolism imaging diagnosis

原始问题：How is DVT diagnosed?
改写后的查询：
DVT diagnosis methods
deep vein thrombosis diagnostic criteria
DVT Wells criteria assessment

## 示例 — 英文领域外问题（不应该改写）
原始问题：How do I configure Nginx reverse proxy?
改写后的查询：
How do I configure Nginx reverse proxy?

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


# ── 知识域过滤辅助 ────────────────────────────────


def _domain_filter(domain: str | None) -> str | None:
    """Milvus JSON filter 表达式（metadata.domain == <domain>）"""
    if not domain:
        return None
    return f'metadata["domain"] == "{domain}"'


def _filter_by_domain(chunks: list[dict[str, Any]], domain: str | None) -> list[dict[str, Any]]:
    """按 metadata.domain 过滤结果（BM25 侧无存储级 filter，用后置过滤）"""
    if not domain:
        return chunks
    return [c for c in chunks if (c.get("metadata") or {}).get("domain") == domain]


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
        vector_store,
        embedding_provider: EmbeddingProvider,
        top_k: int = 20,
        bm25_weight: float = 0.5,
        generator=None,
        enable_rewrite: bool = True,
        enable_reranker: bool = True,
        rerank_top_k: int = 10,
        max_per_doc: int = 2,
        reranker=None,
        bm25_backend: str = "memory",
        bm25_index_dir: str = "lucene_bm25_index",
        rewrite_generator=None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.generator = generator
        self.rewrite_generator = rewrite_generator
        self.enable_rewrite = enable_rewrite if generator else False
        self.enable_reranker = enable_reranker if generator else False
        self.rerank_top_k = rerank_top_k
        self.max_per_doc = max_per_doc
        self._reranker = reranker
        self._bm25 = None
        self._bm25_backend = bm25_backend if generator else "memory"
        self._bm25_index_dir = bm25_index_dir
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_meta: dict[str, dict] = {}
        self._original_query: str = ""
        self._was_rewritten: bool = False
        self._out_of_domain: bool = False

    # ══════════════════════════════════════════════════
    #  主入口
    # ══════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int | None = None, domain: str | None = None) -> list[dict[str, Any]]:
        """三级召回流水线

        1. Rewrite Gate — 规则判断是否需要改写；若初始检索分低也触发
        2. Hybrid Search（向量 + BM25 -> RRF）
        3. Reranker（可选）

        Args:
            domain: 知识域过滤（pe_literature / writing_guidelines 等，
                取 chunk metadata.domain；None 表示不过滤）
        """
        k = top_k or self.top_k

        # ── 无 LLM → 纯向量检索（兼容旧版） ──
        if self.generator is None:
            query_embedding = self.embedding_provider.embed([query], prefix="query: ")[0]
            where = _domain_filter(domain)
            if where is not None:
                results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=k * 2, where=where)
            else:
                results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=k * 2)
            results = _filter_by_domain(results, domain)
            for r in results:
                r.pop("_rrf_score", None)
                r.pop("_retriever", None)
            # 文档级多样性（与完整分支一致）
            diverse = self._diversify_by_doc(results, max_per_doc=self.max_per_doc)
            if len(diverse) < k:
                diverse_ids = {r["id"] for r in diverse}
                for r in results:
                    if r["id"] not in diverse_ids:
                        diverse.append(r)
                        diverse_ids.add(r["id"])
                        if len(diverse) >= k:
                            break
            return diverse[:k]

        # 记录原始 query，供后续区分改写 query
        self._original_query = query

        # ── 阶段 0: Rewrite Gate ──
        needs_rewrite = self._can_rewrite() and self._rewrite_gate(query)
        search_queries = self._rewrite_query(query) if needs_rewrite else [query]

        # 改写后原始 query 也要纳入检索——改写是"辅助"不是"替代"，
        # 防止 3 条改写都偏离时丢失原始表达（术语精确题尤其依赖原文）
        if needs_rewrite and search_queries and search_queries[0] != query:
            search_queries = [query] + search_queries

        # ── 阶段 1: 混合检索 ──
        # 每条 query 的结果单独成段（按序 + 去重），供跨 query RRF 融合
        per_query_results: list[list[dict]] = []
        seen_ids: set = set()
        fetch_k = max(k * 2, 20)

        for sq in search_queries:
            results = self._hybrid_retrieve(sq, fetch_k=fetch_k, domain=domain)
            per_query_results.append(results)
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])

        # 跨 query RRF 二次融合：多条 query 的结果按排名贡献累计排序，
        # 被多个角度同时命中的 chunk 加分（多角度确认 = 更相关）。
        # 原始 query（第 1 条）加权 ×1.5——它是用户真实意图的最准确表达，
        # 防止机器改写偏离时把正确结果压下去（术语精确题尤其如此）。
        if len(per_query_results) > 1:
            rrf_k = 60
            score_map: dict[str, dict] = {}
            for q_idx, q_results in enumerate(per_query_results):
                weight = 1.5 if q_idx == 0 else 1.0  # 原 query 加权
                for rank, r in enumerate(q_results):
                    cid = r["id"]
                    if cid not in score_map:
                        score_map[cid] = {"score": 0.0, "result": r}
                    score_map[cid]["score"] += weight / (rrf_k + rank + 1)
            all_results = [v["result"] for _, v in sorted(score_map.items(), key=lambda x: x[1]["score"], reverse=True)]
        else:
            all_results = per_query_results[0]

        # ── 阶段 1.5: 文档级多样性（服务链路） ──
        # 大文档 chunk 数多，向量/BM25 天然偏向它们 → top-k 常被同一文档霸榜
        # （实测混合 query 的 top10 曾 9/10 来自同一篇论文）。
        # 每文档最多 max_per_doc 条；不足 k 时用被跳过的候选补齐，保证召回不缩水。
        # 注：仅作用于 retrieve()（服务链路）；agentic 评测冻结的 _hybrid_retrieve 不在此列。
        diverse = self._diversify_by_doc(all_results, max_per_doc=self.max_per_doc)
        if len(diverse) < k:
            seen_ids = {r["id"] for r in diverse}
            for r in all_results:
                if r["id"] not in seen_ids:
                    diverse.append(r)
                    seen_ids.add(r["id"])
                    if len(diverse) >= k:
                        break
        all_results = diverse[:k]

        # ── 阶段 2: Reranker ──
        candidates = all_results[: self.rerank_top_k]
        reranked = self._rerank(query, candidates, k) if self._can_rerank() and len(candidates) > k else candidates[:k]

        # 清理内部字段
        for r in reranked:
            r.pop("_rrf_score", None)
            r.pop("_retriever", None)

        return reranked

    # ══════════════════════════════════════════════════
    #  Query Rewriting
    # ══════════════════════════════════════════════════

    def _can_rewrite(self) -> bool:
        """检查是否满足 rewrite 条件（优先用 rewrite_generator）"""
        llm = self.rewrite_generator or self.generator
        return self.enable_rewrite and llm is not None

    @staticmethod
    def _rewrite_gate(query: str) -> bool:
        """Rewrite Gate — 规则门控，判断是否需要调用 LLM 改写

        规则（满足任意一条即触发改写）:
          1. 含中文疑问词/代词：什么、如何、为什么、怎样、哪些、哪个、区别
          2. 含中文医疗/症状复合词：症状、治疗、诊断、检查、方案、预后、机制
          3. 提问长度 > 15 个字符（短问句如"肺栓塞是什么"直接搜）
          4. 含否定或条件逻辑：如果没有、是否、不是、除了
          5. 含英文疑问词：what、how、why、which、when、where、who
          6. 含英文医学关键术语：diagnosis、treatment、symptoms、signs、risk

        返回 True → 走 LLM rewrite；False → 直接检索。
        """
        if len(query) > 15:
            return True
        complex_patterns = [
            # 中文疑问词
            r"(什么|如何|为什么|怎样|哪些|哪个|怎么|可否|是否)",
            # 中文医疗术语
            r"(症状|治疗|诊断|检查|方案|预后|机制|病因|预防)",
            # 中文关系词
            r"(区别|对比|关系|联系|影响|作用)",
            # 中文条件否定
            r"(如果|假如|没有|不是|除了|条件)",
            # 英文疑问词
            r"\b(what|how|why|which|when|where|who|whose|whom)\b",
            # 英文医学关键术语
            r"\b(diagnosis|treatment|symptoms|signs|risk|therapy|imaging|management|prognosis|prevention)\b",
        ]
        return any(re.search(pat, query, re.IGNORECASE) for pat in complex_patterns)

    def _rewrite_query(self, query: str) -> list[str]:
        """调用 LLM 将用户问题改写为检索友好查询

        优先使用 rewrite_generator（专用于改写的小模型），
        未设置时回退到主 generator（DeepSeek 等）。

        返回 1-3 条搜索 query。LLM 失败时回退到 [原始 query]。
        """
        llm = self.rewrite_generator or self.generator
        self._was_rewritten = False
        self._out_of_domain = False
        try:
            response = llm.chat(
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
            # 过滤 LLM 降级占位文本（API 不可用时不应作为改写结果）
            if line.startswith("[LLM 不可用"):
                return None
            if len(line) >= 4:
                lines.append(line)

        return lines if lines else None

    # ══════════════════════════════════════════════════
    #  Hybrid Search（向量 + BM25 → RRF）
    # ══════════════════════════════════════════════════

    def _hybrid_retrieve(self, query: str, fetch_k: int, domain: str | None = None) -> list[dict[str, Any]]:
        """纯混合检索（不包含 rewrite / rerank），可被多条改写 query 重复调用

        Args:
            domain: 知识域过滤（chunk metadata.domain），None 表示不过滤。
                默认参数保持 agentic 冻结代码的调用签名不变。
        """
        # 1. 向量检索（语义）——Milvus JSON filter 下推到存储层
        query_embedding = self.embedding_provider.embed([query], prefix="query: ")[0]
        where = _domain_filter(domain)
        if where is not None:
            vector_results = self.vector_store.similarity_search(
                query_embedding=query_embedding, top_k=fetch_k, where=where
            )
        else:
            vector_results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=fetch_k)
        for r in vector_results:
            r["_retriever"] = "vector"
            r["_vector_score"] = r.get("score", 0.0)

        # 2. BM25 检索（关键词）— 不论是否改写，都走 BM25
        bm25_results = self._bm25_retrieve(query, top_k=fetch_k)
        bm25_results = _filter_by_domain(bm25_results, domain)

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

    @staticmethod
    def _diversify_by_doc(results: list[dict[str, Any]], max_per_doc: int = 2) -> list[dict[str, Any]]:
        """文档级多样性约束：每个源文档最多保留 max_per_doc 条 chunk

        保持原始相对顺序（RRF/向量排序），仅做截断。filename 缺失的 chunk
        视为独立文档（不限制），避免误伤无元数据的结果。
        """
        if max_per_doc <= 0 or len(results) <= 1:
            return list(results)
        counts: dict[str, int] = {}
        out: list[dict[str, Any]] = []
        for r in results:
            fn = (r.get("metadata") or {}).get("filename", "") or ""
            if not fn:
                out.append(r)
                continue
            if counts.get(fn, 0) >= max_per_doc:
                continue
            counts[fn] = counts.get(fn, 0) + 1
            out.append(r)
        return out

    # ══════════════════════════════════════════════════
    #  Reranker
    # ══════════════════════════════════════════════════

    def _can_rerank(self) -> bool:
        """检查是否满足 reranker 条件（cross-encoder 或 LLM）"""
        return self.enable_reranker and (self._reranker is not None or self.generator is not None)

    def _rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Cross-encoder 重排序"""
        if self._reranker is not None and self._reranker.model_ready:
            try:
                return self._reranker.rerank(query, chunks, top_k)
            except Exception:
                pass
        return chunks[:top_k]

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
        self._bm25_meta = {c["id"]: c.get("metadata", {}) for c in all_chunks}
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
        """从 _bm25_meta 缓存查找 metadata（O(1)，避免每次全量拉取）"""
        meta = self._bm25_meta.get(chunk_id)
        if meta is not None:
            return {"metadata": meta}
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

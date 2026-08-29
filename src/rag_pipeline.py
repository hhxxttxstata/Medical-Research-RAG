"""
RAG 管道 — 编排文档加载、切分、Embedding、检索和生成
"""

import os
import re
import time
from typing import Any

from .document_loader import load_and_process_from_dir, load_documents_from_dir
from .embeddings import get_embedding_provider
from .generator import (
    build_quick_prompt,
    build_rag_prompt,
    compute_relevance,
    create_generator,
    create_rewrite_generator,
    validate_citations,
)
from .logger import get_logger
from .milvus_store import MilvusStore
from .retriever import Retriever
from .text_splitter import split_document

_ENT_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _entity_overlap(question: str, chunks: list[dict], window: int = 8, generic_terms: set[str] | None = None) -> float:
    """区分性实体覆盖率：问题中的区分性实体出现在证据文本中的比例（0-1）

    与 agentic_rag 的 _entity_overlap 同语义：英文词（≥3 字符）+ 中文 2-4 字片段
    （排除通用词）。OOD 问题的实体（糖尿病肾病/诺贝尔/Kubernetes）在库内
    证据中覆盖率趋近 0；库内术语题（GPU显存/L3级备份）覆盖率显著 > 0。
    无区分性实体时返回 1.0（不误报）。参数由离线 81 题网格验证：
    thresh=0.08 + window=8 → OOD 16/16 拒、库内 65/65 不误拒。
    """
    if not chunks:
        return 0.0
    generic = generic_terms or set()
    cand = " ".join(c.get("text", "")[:800] for c in chunks[:window]).lower()
    entities = set(_ENT_RE.findall(question.lower()))
    for n in (4, 3, 2):
        for i in range(len(question) - n + 1):
            frag = question[i : i + n]
            if not re.fullmatch(r"[\u4e00-\u9fff]+", frag):
                continue
            if frag in generic:
                continue
            entities.add(frag)
    if not entities:
        return 1.0
    hit = sum(1 for e in entities if e in cand)
    return hit / len(entities)


def _detect_embedding_dim(embedding_provider) -> int:
    try:
        emb = embedding_provider.embed(["warmup"])
        if emb and len(emb) > 0:
            return len(emb[0])
    except Exception:
        pass
    return 768


class RAGPipeline:
    """RAG 系统主管道"""

    def __init__(
        self,
        data_dir: str = "data",
        embedding_provider: str = "local",
        embedding_model: str | None = None,
        top_k: int = 5,
        chunk_min_chars: int = 300,
        chunk_max_chars: int = 500,
        retriever_mode: str = "hybrid",
        enable_rewrite: bool = True,
        enable_reranker: bool = True,
        reranker=None,
        cache_manager=None,
        bm25_backend: str = "memory",
        bm25_index_dir: str = "lucene_bm25_index",
        vector_backend: str = "milvus",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_lite: bool = False,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.top_k = top_k
        self.chunk_min_chars = chunk_min_chars
        self.chunk_max_chars = chunk_max_chars
        self.retriever_mode = retriever_mode
        self._reranker = reranker
        self._cache = cache_manager
        self._bm25_backend = bm25_backend
        self._bm25_index_dir = bm25_index_dir

        collection_name = f"rag_docs_c{chunk_min_chars}_{chunk_max_chars}"
        self.embedding_provider = get_embedding_provider(embedding_provider, embedding_model)

        self.vector_store = MilvusStore(
            collection_name=collection_name,
            dim=_detect_embedding_dim(self.embedding_provider),
            host=milvus_host,
            port=milvus_port,
            use_lite=milvus_lite,
        )

        self.generator = create_generator()
        self.rewrite_generator = create_rewrite_generator()
        if self.rewrite_generator:
            print("  🤖 Query Rewriting 专用模型: Qwen3-4B-Instruct (SiliconFlow)")
        self.logger = get_logger()

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            top_k=top_k,
            generator=self.generator if retriever_mode == "hybrid" else None,
            enable_rewrite=enable_rewrite if retriever_mode == "hybrid" else False,
            enable_reranker=enable_reranker if retriever_mode == "hybrid" else False,
            reranker=reranker,
            bm25_backend=bm25_backend,
            bm25_index_dir=bm25_index_dir,
            rewrite_generator=self.rewrite_generator,
        )

    def close(self):
        self.vector_store.close()

    def initialize_knowledge_base(self, force_reindex: bool = False, use_smart_chunking: bool = True):
        if force_reindex:
            self.vector_store.delete_collection()

        existing_count = self.vector_store.count()
        if existing_count > 0 and not force_reindex:
            print(f"  ✅ 知识库已初始化，现有 {existing_count} 个 Chunk")
            return existing_count

        print("\n" + "=" * 60)
        print(f"  📚 初始化知识库 (chunk: {self.chunk_min_chars}-{self.chunk_max_chars}字)")
        print("=" * 60 + "\n")

        if use_smart_chunking:
            processed_docs = load_and_process_from_dir(self.data_dir)
            if not processed_docs:
                print("  ❌ 未找到支持的文档")
                return 0
            all_chunks = []
            for pdoc in processed_docs:
                all_chunks.extend(pdoc.get("small_chunks", []))
            print(f"  ✅ 共生成 {len(all_chunks)} 个 Small Chunk\n")
        else:
            documents = load_documents_from_dir(self.data_dir)
            if not documents:
                print("  ❌ 未找到支持的文档")
                return 0
            all_chunks = []
            for doc in documents:
                all_chunks.extend(
                    split_document(doc, chunk_min_chars=self.chunk_min_chars, chunk_max_chars=self.chunk_max_chars)
                )
            print(f"  ✅ 共切分为 {len(all_chunks)} 个 Chunk\n")

        if not all_chunks:
            return 0

        texts = [chunk["text"] for chunk in all_chunks]
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_embeddings = self.embedding_provider.embed(texts[i : i + batch_size], prefix="passage: ")
            all_embeddings.extend(batch_embeddings)
            print(f"  📊 已处理 {min(i + batch_size, len(texts))}/{len(texts)}")

        self.vector_store.add_chunks(all_chunks, all_embeddings)
        print(f"  ✅ 成功存入 {len(all_chunks)} 个 Chunk\n")
        print("=" * 60 + "\n")
        return len(all_chunks)

    def _retrieve_and_prepare(self, question: str, k: int, domain: str | None = None) -> dict:
        """共享检索逻辑：检索 → Small-to-Big 展开 → 相关性判断

        Args:
            domain: 知识域过滤（pe_literature / writing_guidelines 等）

        Returns {
            "retrieved_chunks": list[dict],
            "relevance": dict,
            "is_refusal": bool,
            "cache_hit": str,
        }
        """
        retrieved_chunks: list[dict[str, Any]] = []
        cache_hit = "none"

        # Retrieval Cache
        if self._cache and not domain:
            retrieved_chunks = self._cache.retrieval.get(question, k)
            if retrieved_chunks:
                cache_hit = "retrieval"
                print(f"  ⚡ 检索缓存命中 ({len(retrieved_chunks)} chunks)\n")

        if not retrieved_chunks:
            print(f"📡 检索相关文档 (top_k={k}{', domain=' + domain if domain else ''})...")
            retrieved_chunks = self.retriever.retrieve(question, top_k=k, domain=domain)
            print(f"  ✅ 检索到 {len(retrieved_chunks)} 个相关片段\n")
            if self._cache and retrieved_chunks and not domain:
                self._cache.retrieval.set(question, k, retrieved_chunks)

        # Small-to-Big 展开 + parent_content 去重
        expanded = []
        seen_parents = set()
        for c in retrieved_chunks:
            parent_content = c.get("metadata", {}).get("parent_content", "")
            if parent_content:
                if parent_content not in seen_parents:
                    seen_parents.add(parent_content)
                    c["text"] = parent_content
                    c["_expanded"] = True
                    expanded.append(c)
            else:
                expanded.append(c)
        if expanded:
            deduped = len(retrieved_chunks) - len(expanded)
            print(f"  📐 Small-to-Big 展开完毕 (去重 {deduped} 个)\n")
            retrieved_chunks = expanded

        # 相关性判断
        relevance = compute_relevance(question, retrieved_chunks)
        is_refusal = False
        if getattr(self.retriever, "_out_of_domain", False):
            relevance["is_relevant"] = False
            relevance["reason"] = "LLM Query Rewriting 判定为领域外问题"
            is_refusal = True

        # ── OOD 增强门（2026-08，离线 81 题验证：Refusal Acc 0.877 → 1.000，零误拒）──
        # 背景：e5-base 对 OOD 文本语义分虚高（0.82-0.86），原相关性门禁拦不住
        # （OOD 漏拒率 62.5%）。两条零成本规则：
        #   1. ood_rule 命中（领域外关键词规则）且词面重叠不高 → 否决相关性
        #   2. 区分性实体覆盖率 < 0.08（问题实体几乎不在证据中）→ 领域外
        # 参数网格验证：thresh=0.08 + window=8 → OOD 16/16 拒、库内 65/65 不误拒。
        # 注：只读引用 agentic 冻结模块的规则/停用词，不改冻结代码。
        if relevance["is_relevant"]:
            from .agentic_rag import _GENERIC_TERMS, _is_out_of_domain  # 只读引用

            ood_rule = _is_out_of_domain(question)
            ent_ov = _entity_overlap(question, retrieved_chunks, window=8, generic_terms=_GENERIC_TERMS)
            overlap = relevance.get("overlap", 0.0)
            if (ood_rule and overlap < 0.15) or ent_ov < 0.08:
                relevance["is_relevant"] = False
                relevance["reason"] = (
                    f"OOD 增强门: ood_rule={ood_rule} entity_overlap={ent_ov:.2f} "
                    f"word_overlap={overlap:.3f} → 证据与问题无关"
                )
                is_refusal = True

        return {
            "retrieved_chunks": retrieved_chunks,
            "relevance": relevance,
            "is_refusal": is_refusal,
            "cache_hit": cache_hit,
        }

    def query(self, question: str, top_k: int | None = None, domain: str | None = None) -> dict[str, Any]:
        start_time = time.time()
        k = top_k or self.top_k
        error = None
        is_refusal = False
        retrieved_chunks: list[dict[str, Any]] = []
        cache_hit = "none"

        # 1. Answer Cache
        if self._cache and not domain:
            try:
                cached = self._cache.answer.get(question)
                if cached:
                    elapsed = time.time() - start_time
                    cached["cache_hit"] = "answer"
                    cached["elapsed"] = round(elapsed, 2)
                    print("  ⚡ 回答缓存命中\n")
                    return cached
            except Exception:
                pass

        print(f"\n🔍 查询: {question}")
        print(f"{'=' * 60}\n")

        try:
            # 2. 共享检索逻辑
            prep = self._retrieve_and_prepare(question, k, domain=domain)
            retrieved_chunks = prep["retrieved_chunks"]
            relevance = prep["relevance"]
            is_refusal = prep["is_refusal"]
            cache_hit = prep["cache_hit"]

            # 生成结构化回答
            print("🤖 生成回答...")
            prompt_data = build_rag_prompt(question, retrieved_chunks, relevance)
            gen = self.generator.generate_structured(prompt_data, self_reflect=True)
            answer = gen["raw"]
            structured = gen["structured"]
            citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}

            if gen["valid"] and relevance["is_relevant"]:
                source_map = prompt_data[1] if len(prompt_data) >= 2 else {}
                citation_result = validate_citations(answer, source_map)

        except Exception as e:
            error = str(e)
            answer = f"系统错误: {error}"
            retrieved_chunks = []
            structured = {}
            relevance = {"is_relevant": False, "top1_score": 0, "avg_score": 0, "overlap": 0, "reason": ""}
            citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}
            print(f"  ❌ 错误: {error}")

        elapsed = time.time() - start_time

        # OpenTelemetry trace context
        _trace_id = ""
        _span_id = ""
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            ctx = span.get_span_context() if span else None
            if ctx:
                _trace_id = hex(ctx.trace_id)[2:]
                _span_id = hex(ctx.span_id)[2:]
        except Exception:
            pass

        self.logger.log_query(
            question=question,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            elapsed=elapsed,
            error=error,
            is_refusal=is_refusal,
            top_k=k,
            chunk_min_chars=self.chunk_min_chars,
            chunk_max_chars=self.chunk_max_chars,
            relevance=relevance,
            trace_id=_trace_id,
            span_id=_span_id,
        )

        return {
            "question": question,
            "answer": answer,
            "structured": structured,
            "sources": retrieved_chunks,
            "time": elapsed,
            "error": error,
            "is_refusal": is_refusal,
            "relevance": relevance,
            "citation_validation": citation_result,
            "cache_hit": cache_hit,
        }

    def query_stream(self, question: str, top_k: int | None = None, domain: str | None = None):
        """流式查询——yield SSE event dict，每阶段推送进度"""
        start_time = time.time()
        k = top_k or self.top_k

        # 1. Answer Cache check
        if self._cache and not domain:
            try:
                cached = self._cache.answer.get(question)
                if cached:
                    yield {"event": "answer", "data": cached["answer"]}
                    yield {"event": "sources", "data": cached.get("sources", [])}
                    yield {"event": "elapsed", "data": time.time() - start_time}
                    yield {"event": "cache_hit", "data": "answer"}
                    yield {"event": "done", "data": ""}
                    return
            except Exception:
                pass

        yield {"event": "status", "data": "🔍 正在检索医学知识库..."}

        try:
            # 2. 共享检索逻辑
            prep = self._retrieve_and_prepare(question, k, domain=domain)
            retrieved_chunks = prep["retrieved_chunks"]
            relevance = prep["relevance"]
            is_refusal = prep["is_refusal"]

            yield {"event": "status", "data": f"✅ 已找到 {len(retrieved_chunks)} 条相关依据"}
            yield {"event": "sources", "data": retrieved_chunks}

            if is_refusal or not relevance["is_relevant"]:
                yield {"event": "status", "data": "🤖 正在生成回答..."}
                quick_prompt_data = build_quick_prompt(question, retrieved_chunks, relevance)
                gen = self.generator.generate_structured(quick_prompt_data)
                yield {"event": "answer", "data": gen["raw"]}
            else:
                yield {"event": "status", "data": "🤖 正在生成快速回答..."}
                quick_prompt_data = build_quick_prompt(question, retrieved_chunks, relevance)
                gen = self.generator.generate_structured(quick_prompt_data, self_reflect=False)
                yield {"event": "quick_answer", "data": gen["raw"]}

                # 性能（2026-08 优化）：原实现此处再调一次 LLM 生成 verbose 完整回答，
                # 单请求 2 次生成 ≈ 25-60s。改为单次生成：verbose 复用 quick 结果，
                # 前端"展开"即时展示同一答案，LLM 调用减半、延迟近半。
                yield {"event": "verbose_answer", "data": gen["raw"]}

        except Exception as e:
            yield {"event": "error", "data": str(e)}
            return

        elapsed = time.time() - start_time
        yield {"event": "elapsed", "data": elapsed}

        self.logger.log_query(
            question=question,
            retrieved_chunks=retrieved_chunks,
            answer="(streaming)",
            elapsed=elapsed,
            error=None if not is_refusal else "refusal",
            is_refusal=is_refusal,
            top_k=k,
        )

        yield {"event": "done", "data": ""}

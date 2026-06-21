"""
RAG 管道
编排文档加载、切分、Embedding、检索和生成全流程
"""

import os
import time
from typing import Any

from opentelemetry import trace as otel_trace

from .document_loader import load_and_process_from_dir, load_documents_from_dir
from .embeddings import get_embedding_provider
from .generator import (
    build_rag_prompt,
    compute_relevance,
    create_generator,
    validate_citations,
)
from .logger import get_logger
from .monitoring.metrics import record_kb_size, record_request

# ── 可观测性 ────────────────────────────────────────
from .monitoring.tracing import get_current_span_id, get_current_trace_id, get_tracer
from .retriever import HybridRetriever, Retriever
from .text_splitter import split_document
from .vector_store import VectorStore


class RAGPipeline:
    """RAG 系统主管道"""

    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "chroma_db",
        embedding_provider: str = "local",
        embedding_model: str | None = None,
        top_k: int = 3,
        chunk_min_chars: int = 300,
        chunk_max_chars: int = 500,
        retriever_mode: str = "hybrid",
        enable_rewrite: bool = True,
        enable_reranker: bool = False,
        reranker=None,  # CrossEncoderReranker 实例
        cache_manager=None,  # CacheManager 实例
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.persist_dir = os.path.abspath(persist_dir)
        self.top_k = top_k
        self.chunk_min_chars = chunk_min_chars
        self.chunk_max_chars = chunk_max_chars
        self.retriever_mode = retriever_mode
        self._reranker = reranker
        self._cache = cache_manager

        # 根据 chunk size 确定集合名称
        collection_name = f"rag_docs_c{chunk_min_chars}_{chunk_max_chars}"

        # 初始化各组件
        self.embedding_provider = get_embedding_provider(embedding_provider, embedding_model)
        self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)

        self.generator = create_generator()
        self.logger = get_logger()

        # 根据 mode 选择检索器
        if retriever_mode == "hybrid":
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                top_k=top_k,
                generator=self.generator,
                enable_rewrite=enable_rewrite,
                enable_reranker=enable_reranker,
                reranker=reranker,
            )
        else:
            self.retriever = Retriever(
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                top_k=top_k,
            )

    def close(self):
        """释放底层资源（ChromaDB 文件锁等）"""
        self.vector_store.close()

    def initialize_knowledge_base(self, force_reindex: bool = False, use_smart_chunking: bool = True):
        """初始化知识库：加载文档 → 切分 → Embedding → 存入向量数据库

        Args:
            force_reindex: 是否强制重建索引
            use_smart_chunking: 是否使用智能处理管线（元数据+Markdown+Small-to-Big）
                                False = 使用旧版 text_splitter
        """
        print("\n" + "=" * 60)
        print(f"  📚 开始初始化知识库 (chunk: {self.chunk_min_chars}-{self.chunk_max_chars}字)")
        print("=" * 60 + "\n")

        if force_reindex:
            self.vector_store.delete_collection()

        # 检查是否已有数据
        existing_count = self.vector_store.count()
        if existing_count > 0 and not force_reindex:
            print(f"  ✅ 知识库已初始化，现有 {existing_count} 个 Chunk")
            print("  💡 如需重新索引，请使用 --rebuild 或 --chunk-size 参数")
            return existing_count

        # 1. 加载文档（智能管线或旧版模式）
        print("📖 步骤 1/4: 加载文档...")
        if use_smart_chunking:
            print("  🧠 启用智能管线: 元数据提取 + Markdown转换 + Small-to-Big切分")
            processed_docs = load_and_process_from_dir(self.data_dir)
            print(f"  ✅ 共处理 {len(processed_docs)} 个文档\n")

            if not processed_docs:
                print("  ❌ 未找到支持的文档")
                return 0

            # 收集所有 small chunks 作为检索向量
            all_chunks = []
            for pdoc in processed_docs:
                all_chunks.extend(pdoc.get("small_chunks", []))
            print(f"  ✅ 共生成 {len(all_chunks)} 个 Small Chunk（用于向量检索）\n")

        else:
            # 旧版管线
            documents = load_documents_from_dir(self.data_dir)
            print(f"  ✅ 共加载 {len(documents)} 个文档\n")
            if not documents:
                print("  ❌ 未找到支持的文档（支持格式: PDF, MD, TXT）")
                return 0
            print(f"✂️  步骤 2/4: 文本切分 (chunk: {self.chunk_min_chars}-{self.chunk_max_chars}字)...")
            all_chunks = []
            for doc in documents:
                chunks = split_document(
                    doc,
                    chunk_min_chars=self.chunk_min_chars,
                    chunk_max_chars=self.chunk_max_chars,
                )
                all_chunks.extend(chunks)
            print(f"  ✅ 共切分为 {len(all_chunks)} 个 Chunk\n")

        if not all_chunks:
            print("  ❌ 切分结果为空")
            return 0

        # 3. 生成 Embedding（e5 模型需要 "passage: " 前缀）
        print("🧬 步骤 3/4: 生成 Embedding...")
        texts = [chunk["text"] for chunk in all_chunks]
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_provider.embed(batch_texts, prefix="passage: ")
            all_embeddings.extend(batch_embeddings)
            print(f"  📊 已处理 {min(i + batch_size, len(texts))}/{len(texts)}")

        print(f"  ✅ Embedding 完成，维度: {len(all_embeddings[0])}\n")

        # 4. 存入向量数据库
        print("💾 步骤 4/4: 存入向量数据库...")
        self.vector_store.add_chunks(all_chunks, all_embeddings)

        final_count = self.vector_store.count()
        print(f"  ✅ 成功存入 {final_count} 个 Chunk\n")
        print("=" * 60)
        print("  🎉 知识库初始化完成！")
        print("=" * 60 + "\n")

        # 记录指标
        record_kb_size(final_count)

        return final_count

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        执行完整的 RAG 查询（带多级缓存）

        缓存流程:
          1. AnswerCache 精确+语义匹配 → 命中直接返回
          2. RetrievalCache 精确匹配 → 命中跳过检索
          3. 检索 + Cross-encoder/LLM Reranker
          4. LLM 生成 → AnswerCache.set

        返回:
            {
                "question": str,
                "answer": str,
                "sources": [...],
                "time": float,
                "error": str,
                "is_refusal": bool,
                "relevance": {...},
                "citation_validation": {...},
                "cache_hit": str,   # "answer" | "retrieval" | "none"
            }
        """
        start_time = time.time()
        k = top_k or self.top_k
        error = None
        is_refusal = False
        retrieved_chunks: list[dict[str, Any]] = []
        cache_hit = "none"

        # ── 1. Answer Cache ──
        if self._cache:
            try:
                cached = self._cache.answer.get(question)
                if cached:
                    elapsed = time.time() - start_time
                    cached["cache_hit"] = "answer"
                    cached["elapsed"] = round(elapsed, 2)
                    print("  ⚡ 回答缓存命中 (annswer)\n")
                    return cached
            except Exception:
                pass

        # ── 追踪 span ──
        tracer = get_tracer()
        span_ctx = tracer.start_as_current_span("rag_pipeline.query")
        query_span = span_ctx.__enter__()
        query_span.set_attribute("question_length", len(question))
        query_span.set_attribute("top_k", k)

        print(f"\n🔍 查询: {question}")
        print(f"{'=' * 60}\n")

        try:
            # ── 2. Retrieval Cache ──
            if self._cache:
                retrieved_chunks = self._cache.retrieval.get(question, k)
                if retrieved_chunks:
                    cache_hit = "retrieval"
                    print(f"  ⚡ 检索缓存命中 ({len(retrieved_chunks)} chunks)\n")

            if not retrieved_chunks:
                # 1. 检索
                print(f"📡 步骤 1/2: 检索相关文档 (top_k={k})...")
                with tracer.start_as_current_span("pipeline.retrieval") as span:
                    retrieved_chunks = self.retriever.retrieve(question, top_k=k)
                    span.set_attribute("chunk_count", len(retrieved_chunks))
                print(f"  ✅ 检索到 {len(retrieved_chunks)} 个相关片段\n")

                # 写回缓存
                if self._cache and retrieved_chunks:
                    self._cache.retrieval.set(question, k, retrieved_chunks)

            # 1.2 Small-to-Big 展开：将 small chunk 替换为对应的 parent chunk
            expanded = []
            seen_parents = set()
            for c in retrieved_chunks:
                parent_content = c.get("metadata", {}).get("parent_content", "")
                if parent_content:
                    # 用 parent 去重：多个 small chunk 映射到同一个 parent 时只保留一份
                    parent_key = str(hash(parent_content))
                    if parent_key not in seen_parents:
                        seen_parents.add(parent_key)
                        c["text"] = parent_content
                        c["_expanded"] = True
                        expanded.append(c)
                else:
                    expanded.append(c)
            if expanded:
                retained = len(expanded)
                deduped = len(retrieved_chunks) - retained
                print(f"  📐 Small-to-Big: 展开为 parent chunk (去重 {deduped} 个重复), 保留 {retained} 个完整上下文\n")
                retrieved_chunks = expanded

            # 1.5 计算相关性（仅用于 Prompt 策略，不再硬拒答）
            relevance = compute_relevance(question, retrieved_chunks)

            # LLM 改写已判断为领域外 → 覆盖相关性为不相关
            if hasattr(self.retriever, "_out_of_domain") and self.retriever._out_of_domain:
                relevance["is_relevant"] = False
                relevance["reason"] = f"LLM Query Rewriting 判定为领域外问题（覆盖：{relevance['reason']}）"
                is_refusal = True
                print(
                    f"  📊 相关性判断: ⚠️ LLM 判定为领域外，启用拒答 "
                    f"(top1={relevance['top1_score']:.3f}, overlap={relevance['overlap']:.3f})"
                )

            # 不再触发硬拒答 — 改为混合模式，让 LLM 综合自身知识回答

            # 2. 构建 Prompt 并生成回答
            print("🤖 步骤 2/2: 生成回答（混合模式：参考文档 + 自身知识）...")
            prompt_data = build_rag_prompt(question, retrieved_chunks, relevance)
            with tracer.start_as_current_span("pipeline.generation") as span:
                answer = self.generator.generate(prompt_data)
                span.set_attribute("answer_length", len(answer))
                span.set_attribute("is_refusal", is_refusal)

            # 3. 引用验证（只验证相关性较高时的引用）
            if relevance["is_relevant"] and retrieved_chunks:
                source_map = prompt_data[1] if len(prompt_data) >= 2 else {}
                citation_result = validate_citations(answer, source_map)
                if citation_result["has_invalid_citations"]:
                    print(f"  ⚠️ 引用验证: 发现无效引用 {citation_result['cited_invalid']}")
            else:
                citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}

        except Exception as e:
            error = str(e)
            answer = f"系统错误: {error}"
            retrieved_chunks = []
            relevance = {"is_relevant": False, "top1_score": 0, "avg_score": 0, "overlap": 0, "reason": ""}
            citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}
            query_span.record_exception(e)
            query_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
            print(f"  ❌ 错误: {error}")

        elapsed = time.time() - start_time
        query_span.set_attribute("duration_ms", round(elapsed * 1000, 1))
        query_span.set_attribute("has_error", error is not None)

        # 在 span 关闭前提取 trace context
        trace_id = get_current_trace_id()
        span_id = get_current_span_id()

        span_ctx.__exit__(None, None, None)

        # 记录指标
        status = 500 if error else (200 if not is_refusal else 204)
        record_request("rag_query", status, elapsed)
        scores = [c.get("score", 0) for c in retrieved_chunks]
        from .monitoring.metrics import record_retrieval as _record_retrieval

        if scores:
            _record_retrieval(scores)

        # 记录日志（含 trace_id）
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
            trace_id=trace_id,
            span_id=span_id,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks,
            "time": elapsed,
            "error": error,
            "is_refusal": is_refusal,
            "relevance": relevance,
            "citation_validation": citation_result,
            "cache_hit": cache_hit,
        }

    def print_result(self, result: dict[str, Any]):
        """格式化打印查询结果（企业知识库风格）"""
        print("\n" + "=" * 70)
        print("  📋 查询结果")
        print("=" * 70)

        print(f"\n❓ 问题: {result['question']}\n")

        if result.get("is_refusal"):
            print("🚫 **拒答**: 知识库中未找到相关依据，已拒绝回答")
            print()

        print("💡 回答:")
        print("=" * 70)
        print(result["answer"])
        print("=" * 70)

        # 引用溯源
        if result.get("sources") and not result.get("is_refusal"):
            print(f"\n📚 引用来源 ({len(result['sources'])} 个):")
            print("-" * 70)
            for i, src in enumerate(result["sources"], 1):
                meta = src["metadata"]
                filename = meta.get("filename", "未知")
                page = meta.get("page", "")
                score = src["score"]
                text_preview = src["text"][:120].replace("\n", " ")

                print(f"\n  [{i}] 📄 {filename}")
                if page:
                    print(f"      📄 页码: {page}")
                if src.get("metadata", {}).get("paragraph_start"):
                    print(f"      📄 段落: {meta.get('paragraph_start')}-{meta.get('paragraph_end')}")
                print(f"      📊 相似度: {score:.3f}")
                print(f"      📝 原文: {text_preview}...")

            # 引用验证信息
            cv = result.get("citation_validation", {})
            if cv and cv.get("cited_valid"):
                print(
                    f"\n  ✅ 引用验证: 回答中使用了 {len(cv['cited_valid'])}/{len(cv.get('unused', []) or []) + len(cv['cited_valid'])} 个可用来源"
                )
                print(f"     引用编号: {', '.join(cv['cited_valid'])}")

        # 相关性信息
        relevance = result.get("relevance", {})
        if relevance:
            print(
                f"\n📊 检索质量: 最高相似度={relevance.get('top1_score', 0):.3f}, "
                f"平均相似度={relevance.get('avg_score', 0):.3f}, "
                f"文本重叠率={relevance.get('overlap', 0):.3f}"
            )

        if result.get("error"):
            print(f"\n❌ 错误: {result['error']}")

        print(f"\n⏱️  耗时: {result['time']:.2f} 秒")
        print("=" * 70 + "\n")

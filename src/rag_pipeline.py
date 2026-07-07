"""
RAG 管道 — 编排文档加载、切分、Embedding、检索和生成
"""

import os
import time
from typing import Any

from .document_loader import load_and_process_from_dir, load_documents_from_dir
from .embeddings import get_embedding_provider
from .generator import build_rag_prompt, compute_relevance, create_generator, validate_citations
from .logger import get_logger
from .retriever import Retriever
from .text_splitter import split_document
from .vector_store import VectorStore, create_vector_store


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
        persist_dir: str = "chroma_db",
        embedding_provider: str = "local",
        embedding_model: str | None = None,
        top_k: int = 3,
        chunk_min_chars: int = 300,
        chunk_max_chars: int = 500,
        retriever_mode: str = "hybrid",
        enable_rewrite: bool = True,
        enable_reranker: bool = True,
        reranker=None,
        cache_manager=None,
        bm25_backend: str = "memory",
        bm25_index_dir: str = "lucene_bm25_index",
        vector_backend: str = "chroma",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_lite: bool = False,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.persist_dir = os.path.abspath(persist_dir)
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

        if vector_backend == "milvus":
            self.vector_store = create_vector_store(
                backend="milvus",
                collection_name=collection_name,
                dim=_detect_embedding_dim(self.embedding_provider),
                host=milvus_host,
                port=milvus_port,
                use_lite=milvus_lite,
            )
        else:
            self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)

        self.generator = create_generator()
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
                all_chunks.extend(split_document(doc, chunk_min_chars=self.chunk_min_chars, chunk_max_chars=self.chunk_max_chars))
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
        final_count = self.vector_store.count()
        print(f"  ✅ 成功存入 {final_count} 个 Chunk\n")
        print("=" * 60 + "\n")
        return final_count

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        start_time = time.time()
        k = top_k or self.top_k
        error = None
        is_refusal = False
        retrieved_chunks: list[dict[str, Any]] = []
        cache_hit = "none"

        # 1. Answer Cache
        if self._cache:
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
            # 2. Retrieval Cache
            if self._cache:
                retrieved_chunks = self._cache.retrieval.get(question, k)
                if retrieved_chunks:
                    cache_hit = "retrieval"
                    print(f"  ⚡ 检索缓存命中 ({len(retrieved_chunks)} chunks)\n")

            if not retrieved_chunks:
                print(f"📡 检索相关文档 (top_k={k})...")
                retrieved_chunks = self.retriever.retrieve(question, top_k=k)
                print(f"  ✅ 检索到 {len(retrieved_chunks)} 个相关片段\n")
                if self._cache and retrieved_chunks:
                    self._cache.retrieval.set(question, k, retrieved_chunks)

            # Small-to-Big 展开
            expanded = []
            seen_parents = set()
            for c in retrieved_chunks:
                parent_content = c.get("metadata", {}).get("parent_content", "")
                if parent_content:
                    parent_key = str(hash(parent_content))
                    if parent_key not in seen_parents:
                        seen_parents.add(parent_key)
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

            if getattr(self.retriever, "_out_of_domain", False):
                relevance["is_relevant"] = False
                relevance["reason"] = f"LLM Query Rewriting 判定为领域外问题"
                is_refusal = True

            # 生成回答
            print("🤖 生成回答...")
            prompt_data = build_rag_prompt(question, retrieved_chunks, relevance)
            answer = self.generator.generate(prompt_data)

            # 引用验证
            if relevance["is_relevant"] and retrieved_chunks:
                source_map = prompt_data[1] if len(prompt_data) >= 2 else {}
                citation_result = validate_citations(answer, source_map)
            else:
                citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}

        except Exception as e:
            error = str(e)
            answer = f"系统错误: {error}"
            retrieved_chunks = []
            relevance = {"is_relevant": False, "top1_score": 0, "avg_score": 0, "overlap": 0, "reason": ""}
            citation_result = {"cited_valid": [], "cited_invalid": [], "has_invalid_citations": False}
            print(f"  ❌ 错误: {error}")

        elapsed = time.time() - start_time

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

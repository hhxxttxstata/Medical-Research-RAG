"""
Step 3 诊断 — 48 个 Original Miss 的 Failure 分类

对 Original Top10 miss 的题，检索深度放大到 Top100（离线诊断，不影响生产），
记录 Gold 在 Dense / BM25 / Original RRF / 各改写 query 下的 rank@100。

目标: 区分 5 类 failure:
  A. Corpus Absence — Gold 不在库里 (rank = None 全部)
  B. Retrieval Miss — Gold 在 rank 11~100（最值得 Query Transformation）
  C. Lexical / Cross-lingual mismatch — Dense 找不到但 BM25 找到，或反之
  D. Chunking failure — 有答案但证据被切碎
  E. Multi-hop — 需要多文档/多 chunk

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/rewrite_failure_diagnosis.py

产出: eval_results/rewrite_failure_diagnosis_<timestamp>.json
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

from eval.test_questions import get_test_questions  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.retriever import Retriever  # noqa: E402

DEPTH = 100  # 诊断深度


def gold_rank(sources: list[dict], expected: str) -> int | None:
    """Gold 在结果中的 rank（1-based），不在则 None"""
    if not expected:
        return None
    expected_base = expected.rsplit(".", 1)[0]
    for i, s in enumerate(sources):
        fn = s["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            return i + 1
    return None


def main():
    print("=" * 70)
    print("  🔬 Step 3: 48 个 Original Miss 的 Failure 分类 (Top100 诊断)")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)}）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=10,
        generator=generator,
        enable_rewrite=True,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )

    questions = get_test_questions()
    miss_cases = []
    total_miss = 0

    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")

        # Original Top10 命中 → 跳过
        orig_top10 = retriever._hybrid_retrieve(question, fetch_k=10)
        if gold_rank(orig_top10, expected) is not None:
            continue

        # 触发 Gate 才算（与 step1/2 同口径）
        if not (retriever._can_rewrite() and retriever._rewrite_gate(question)):
            continue
        total_miss += 1

        case = {"question": question, "expected": expected, "category": q.get("category", "")}

        # 1. Dense rank@100
        q_emb = provider.embed([question], prefix="query: ")[0]
        dense = store.similarity_search(query_embedding=q_emb, top_k=DEPTH)
        case["dense_rank"] = gold_rank(dense, expected)

        # 2. BM25 rank@100
        bm25_res = bm25.search(question, top_k=DEPTH)
        case["bm25_rank"] = gold_rank(bm25_res, expected)

        # 3. Original RRF rank@100（Dense+BM25 融合，取前 100）
        fused = retriever._rrf_fusion(dense, bm25_res, top_k=DEPTH)
        case["rrf_rank"] = gold_rank(fused, expected)

        # 4. 各改写 query 的 Dense+BM25 rank@100
        rewritten = retriever._rewrite_query(question)
        if rewritten and rewritten[0] != question:
            for qi, rq in enumerate(rewritten, 1):
                rq_emb = provider.embed([rq], prefix="query: ")[0]
                rd = store.similarity_search(query_embedding=rq_emb, top_k=DEPTH)
                rb = bm25.search(rq, top_k=DEPTH)
                rf = retriever._rrf_fusion(rd, rb, top_k=DEPTH)
                case[f"rewrite{qi}_dense_rank"] = gold_rank(rd, expected)
                case[f"rewrite{qi}_bm25_rank"] = gold_rank(rb, expected)
                case[f"rewrite{qi}_rrf_rank"] = gold_rank(rf, expected)
        else:
            case["rewrite_none"] = True

        miss_cases.append(case)
        if total_miss % 8 == 0:
            print(f"    ...{total_miss} miss 已诊断", flush=True)

    elapsed = time.time() - t0

    # ── 分类统计 ──
    A_corpus_absent = [
        c
        for c in miss_cases
        if all(c.get(f"rewrite{i}_rrf_rank") is None for i in (1, 2, 3)) and c.get("rrf_rank") is None
    ]
    B_retrieval = [c for c in miss_cases if c.get("rrf_rank") is not None and 11 <= c["rrf_rank"] <= 100]
    # C: lexical mismatch — dense 找不到(>100)但 bm25 找到(<=100)，或反之
    C_lexical = [
        c
        for c in miss_cases
        if (c.get("dense_rank") is None and c.get("bm25_rank") is not None and c.get("bm25_rank") <= 100)
        or (c.get("dense_rank") is not None and c.get("dense_rank") <= 100 and c.get("bm25_rank") is None)
    ]
    # E: multi-hop 类（近似：问题含"和/与/对比/区别/流程"等并列结构）
    E_multihop = [
        c
        for c in miss_cases
        if any(k in c["question"] for k in ["和", "与", "对比", "区别", "流程", "路径", "关系", "作用"])
    ]
    # D: 其余（无明确分类信号）
    known = set(id(c) for c in A_corpus_absent + B_retrieval + C_lexical)
    D_chunking = [c for c in miss_cases if id(c) not in known]

    print("\n" + "=" * 70)
    print("  📊 48 Miss 分类结果")
    print("=" * 70)
    print(f"  Total miss (Original Top10 miss + Gate 触发) = {total_miss}")
    print(f"  A. Corpus Absence (库里没有)       = {len(A_corpus_absent)}")
    print(f"  B. Retrieval Miss (rank 11~100)    = {len(B_retrieval)}")
    print(f"  C. Lexical mismatch (dense/bm25 单边) = {len(C_lexical)}")
    print(f"  E. Multi-hop 疑似 (并列结构)        = {len(E_multihop)}")
    print(f"  D. 其他 / Chunking                 = {len(D_chunking)}")

    print("\n  📋 B 类 (Retrieval Miss, Gold 在 rank 11~100) 明细:")
    for c in B_retrieval:
        print(
            f"    dense={str(c.get('dense_rank')):>4} bm25={str(c.get('bm25_rank')):>4} rrf={str(c.get('rrf_rank')):>4} "
            f"| r1={str(c.get('rewrite1_rrf_rank')):>4} r2={str(c.get('rewrite2_rrf_rank')):>4} r3={str(c.get('rewrite3_rrf_rank')):>4}"
            f" | {c['question'][:38]}"
        )
    print("\n  📋 C 类 (Lexical mismatch) 明细:")
    for c in C_lexical:
        print(f"    dense={str(c.get('dense_rank')):>4} bm25={str(c.get('bm25_rank')):>4} | {c['question'][:45]}")
    print("\n  📋 E 类 (Multi-hop 疑似) 明细:")
    for c in E_multihop:
        print(f"    rrf={str(c.get('rrf_rank')):>4} | {c['question'][:50]}")

    out = OUT_DIR / f"rewrite_failure_diagnosis_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_miss": total_miss,
                "A_corpus_absent": A_corpus_absent,
                "B_retrieval_miss": B_retrieval,
                "C_lexical": C_lexical,
                "E_multihop": E_multihop,
                "D_other": D_chunking,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

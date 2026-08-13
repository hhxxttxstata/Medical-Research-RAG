"""
RRF 融合 k 参数消融实验 — 对比不同 k 的 Hybrid 检索效果

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/rrf_k_ablation.py

设计:
  - 复用生产 Milvus 集合 rag_docs_c300_500（已嵌入，不重建索引）
  - 复用生产 BM25 索引目录 lucene_bm25_index（若不存在则从数据重建）
  - 仅变换 _rrf_fusion 的 k 参数（16/32/60/128/256），跑 81 题评测
  - hit 判定逻辑与 evaluate.py 完全一致（expected_doc 文件名匹配）

产出: eval_results/rrf_k_ablation_<timestamp>.json
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

from eval.metrics import compute_all_metrics  # noqa: E402
from eval.test_questions import get_test_questions  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402

# k 候选值：论文默认 60，两侧对称取 16/32/60/128/256
K_VALUES = [16, 32, 60, 128, 256]
FETCH_K = 20  # 与 evaluate.py 一致的双路召回数
TOP_K = 10  # 融合后取前 10


def rrf_fuse(vec_results: list[dict], bm25_results: list[dict], top_k: int, k: int) -> list[dict]:
    """复刻 retriever._rrf_fusion，仅 k 可调（逻辑完全一致）"""
    vector_rank = {r["id"]: i + 1 for i, r in enumerate(vec_results)}
    bm25_rank = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}
    all_ids = set(vector_rank.keys()) | set(bm25_rank.keys())
    rrf_scores = {}
    id_to_result = {}
    vector_scores = {}
    for r in vec_results:
        vs = r.get("_vector_score")
        vector_scores[r["id"]] = vs if vs is not None else r.get("score", 0.0)
    for r in vec_results:
        id_to_result[r["id"]] = r
    for r in bm25_results:
        if r["id"] not in id_to_result:
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
                "_vector_score": round(vector_scores.get(cid, 0.0), 4),
                "_rrf_score": rrf_scores[cid],
                "_retriever": "hybrid",
            }
        )
    return results


def run_k(k: int, store, bm25, provider, questions) -> dict:
    """对指定 k 跑 81 题评测"""
    records = []
    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")
        q_emb = provider.embed([question], prefix="query: ")[0]
        vec_results = store.similarity_search(query_embedding=q_emb, top_k=FETCH_K)
        for r in vec_results:
            r["_vector_score"] = r.get("score", 0.0)
        bm25_results = bm25.search(question, top_k=FETCH_K)

        if bm25_results and vec_results:
            sources = rrf_fuse(vec_results, bm25_results, TOP_K, k)
        elif vec_results:
            sources = vec_results[:TOP_K]
        else:
            sources = []

        expected_hit = False
        if expected and sources:
            expected_base = expected.rsplit(".", 1)[0]
            expected_hit = any(
                expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
                for s in sources
            )

        records.append(
            {
                "question": question,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "expected_doc": expected,
                "expected_hit": expected_hit,
                "num_retrieved": len(sources),
                "top_score": round(sources[0].get("_vector_score") or sources[0].get("score", 0), 4) if sources else 0,
            }
        )
    metrics = compute_all_metrics(records)
    return {"k": k, "metrics": metrics, "records": records, "elapsed": round(time.time() - t0, 1)}


def main():
    print("=" * 70)
    print("  🔬 RRF k 参数消融实验")
    print("=" * 70, flush=True)

    from src.embeddings import get_embedding_provider

    provider = get_embedding_provider("local")
    provider.warmup()

    # 生产 Milvus 集合（已嵌入 4256 chunks，不重建）
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 集合: rag_docs_c300_500 (chunks: {store.count()})", flush=True)

    # 生产 BM25 索引（从集合重建，确保与向量库 chunks 一一对应）
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)} 文档）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25 索引文档数: {bm25.get_total_docs()}", flush=True)

    questions = get_test_questions()
    print(f"  📝 测试题: {len(questions)}\n", flush=True)

    results = []
    for k in K_VALUES:
        print(f"  🔬 k={k} ...", flush=True)
        try:
            r = run_k(k, store, bm25, provider, questions)
            results.append(r)
            m = r["metrics"]["overall"]
            print(
                f"    ✅ k={k}: hit_rate={m['hit_rate']:.1%} mrr={m['mrr']:.3f} "
                f"ndcg={m['ndcg_at_5']:.3f} ({r['elapsed']}s)",
                flush=True,
            )
        except Exception as e:
            print(f"    ❌ k={k} 失败: {e}")
            import traceback

            traceback.print_exc()

    # 汇总对比表
    print("\n" + "=" * 70)
    print("  📊 RRF k 参数消融对比")
    print("=" * 70)
    header = f"{'k':>6} {'Hit Rate':>10} {'MRR':>8} {'NDCG@5':>9} {'语义分':>8}"
    print(header)
    print("-" * 50)
    for r in results:
        m = r["metrics"]["overall"]
        print(
            f"{r['k']:>6} {m['hit_rate']:>9.1%} {m['mrr']:>8.3f} "
            f"{m['ndcg_at_5']:>9.3f} {r['metrics'].get('semantic_score', 0):>8.4f}"
        )
    print("-" * 50)

    out = OUT_DIR / f"rrf_k_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k_values": K_VALUES, "results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

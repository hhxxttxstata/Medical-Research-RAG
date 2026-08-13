"""
检索链路组件消融 — 对比 rewrite / hybrid / reranker 各组件贡献

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/pipeline_ablation.py

设计:
  - 复用生产 Milvus 集合 rag_docs_c300_500（不重建索引）
  - 复用生产 BM25 索引目录（不存在则重建）
  - 4 个变体: full / no_rewrite / no_reranker / vector_only
  - 81 题，hit 判定与 evaluate.py 一致

产出: eval_results/pipeline_ablation_<timestamp>.json
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
from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

FETCH_K = 20
TOP_K = 10  # 可通过命令行 --top-k 覆盖


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="检索链路组件消融")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="最终输出 top_k（默认 10）")
    return parser.parse_args()


def build_retriever(store, provider, generator, reranker, mode: str, enable_rewrite: bool, enable_reranker: bool):
    """构造指定变体的 Retriever（复用同一 Milvus/BM25 索引）"""
    is_hybrid = mode == "hybrid"
    return Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=TOP_K,
        generator=generator if is_hybrid else None,
        enable_rewrite=enable_rewrite if is_hybrid else False,
        enable_reranker=enable_reranker if is_hybrid else False,
        reranker=reranker if enable_reranker else None,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
        rewrite_generator=None,  # 用主 generator（生产配置）
    )


def run_variant(name: str, retriever, provider, questions) -> dict:
    """对指定变体跑 81 题检索评测"""
    records = []
    t0 = time.time()
    n_rewrites = 0
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")
        results = retriever.retrieve(question, top_k=TOP_K)
        if getattr(retriever, "_was_rewritten", False):
            n_rewrites += 1

        expected_hit = False
        if expected and results:
            expected_base = expected.rsplit(".", 1)[0]
            expected_hit = any(
                expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
                for s in results
            )

        records.append(
            {
                "question": question,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "expected_doc": expected,
                "expected_hit": expected_hit,
                "num_retrieved": len(results),
            }
        )
    metrics = compute_all_metrics(records)
    return {
        "variant": name,
        "metrics": metrics,
        "records": records,
        "elapsed": round(time.time() - t0, 1),
        "rewrite_count": n_rewrites,
    }


def main():
    args = parse_args()
    global TOP_K
    TOP_K = args.top_k

    print("=" * 70)
    print(f"  🔬 检索链路组件消融（生产栈, top_k={TOP_K}）")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: rag_docs_c300_500 ({store.count()} chunks)", flush=True)

    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)} 文档）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()} 文档", flush=True)

    reranker = CrossEncoderReranker()
    questions = get_test_questions()
    print(f"  📝 测试题: {len(questions)}\n", flush=True)

    variants = [
        ("full (rewrite+hybrid+rerank)", "hybrid", True, True),
        ("no_rewrite (hybrid+rerank)", "hybrid", False, True),
        ("no_reranker (rewrite+hybrid)", "hybrid", True, False),
        ("vector_only (无所有组件)", "vector", False, False),
    ]

    results = []
    for name, mode, rw, rk in variants:
        print(f"  🔬 {name} ...", flush=True)
        retriever = build_retriever(store, provider, generator, reranker, mode, rw, rk)
        try:
            r = run_variant(name, retriever, provider, questions)
            results.append(r)
            m = r["metrics"]["overall"]
            print(
                f"    ✅ hit={m['hit_rate']:.1%} mrr={m['mrr']:.3f} ndcg={m['ndcg_at_5']:.3f} "
                f"rewrites={r['rewrite_count']} ({r['elapsed']}s)",
                flush=True,
            )
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  📊 检索链路组件消融对比")
    print("=" * 70)
    header = f"{'变体':<34} {'Hit Rate':>10} {'MRR':>8} {'NDCG@5':>9} {'重写数':>6}"
    print(header)
    print("-" * 72)
    for r in results:
        m = r["metrics"]["overall"]
        print(
            f"{r['variant']:<34} {m['hit_rate']:>9.1%} {m['mrr']:>8.3f} "
            f"{m['ndcg_at_5']:>9.3f} {r.get('rewrite_count', 0):>6}"
        )
    print("-" * 72)

    out = OUT_DIR / f"pipeline_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

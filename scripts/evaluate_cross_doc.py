"""
Cross-doc 检索评测 — 13 道跨文档综合题

背景：evaluate.py 的 Hit/MRR/NDCG 只覆盖 exact_match（28 题），13 道 cross_doc
题由文档级 gold 标注（tests/cross_doc_gold.json，每题 2-3 个答案所需文档）评测。

本脚本：
  1. 读取 tests/cross_doc_gold.json（文档级 gold 标注）
  2. 用服务配置（rewrite/rerank 关、文档多样性 max_per_doc=2、top_k=8）检索
  3. 计算指标：
       - Doc Recall@k     每题命中 gold docs 比例的平均
       - Coverage@k       平均每题命中的 gold docs 数量
       - Doc MRR@k        每题第一个 gold doc 的 rank 倒数平均
       - Full-Coverage 率 全部 gold docs 命中的题占比
       - AtLeast1 命中率  至少命中 1 个 gold doc 的题占比
  4. 逐题明细（命中/遗漏）写入 eval_results/cross_doc_eval_<ts>.json

用法:
    python scripts/evaluate_cross_doc.py            # 完整评测（串行独占 Milvus）
    python scripts/evaluate_cross_doc.py --top-k 10
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.test_questions import get_test_questions  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

TOP_K = int(os.getenv("CROSS_DOC_TOP_K", "8"))


def load_gold() -> dict:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "cross_doc_gold.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def main():
    questions = [q for q in get_test_questions() if q.get("category") == "cross_doc"]
    gold = load_gold()
    print(f"cross_doc 题数: {len(questions)} | gold 标注: {len(gold)} 题")

    # Milvus 后端：默认 Milvus Lite；设 MILVUS_LITE=false 用 Docker Milvus
    use_lite = os.getenv("MILVUS_LITE", "true").lower() == "true"
    pipeline = RAGPipeline(
        data_dir="data",
        top_k=TOP_K,
        enable_rewrite=False,
        enable_reranker=False,
        milvus_lite=use_lite,
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
        vector_backend="milvus",
    )
    pipeline.retriever._ensure_bm25_index()

    records = []
    t0 = time.time()
    for q in questions:
        qid = q["id"]
        expected_docs = gold.get(qid, [])
        sources = pipeline.retriever.retrieve(q["question"], top_k=TOP_K)
        hit_fns = set()
        ranks = {}
        for i, s in enumerate(sources, start=1):
            fn = (s.get("metadata") or {}).get("filename", "") or ""
            for ed in expected_docs:
                if fn == ed or fn == ed.rsplit(".", 1)[0]:
                    hit_fns.add(ed)
                    ranks.setdefault(ed, i)
        records.append(
            {
                "id": qid,
                "question": q["question"],
                "gold_docs": expected_docs,
                "hit_docs": sorted(hit_fns),
                "missed_docs": sorted(set(expected_docs) - hit_fns),
                "ranks": {ed: ranks[ed] for ed in sorted(ranks)},
                "num_retrieved": len(sources),
            }
        )

    n = len(records)
    total_gold = sum(len(r["gold_docs"]) for r in records)
    total_hit = sum(len(r["hit_docs"]) for r in records)
    recall = sum(len(r["hit_docs"]) / len(r["gold_docs"]) for r in records) / n
    coverage = total_hit / n
    mrr = 0.0
    for r in records:
        first = min(r["ranks"].values()) if r["ranks"] else None
        if first:
            mrr += 1.0 / first
    mrr /= n
    full = sum(1 for r in records if set(r["hit_docs"]) == set(r["gold_docs"]))
    atleast1 = sum(1 for r in records if r["hit_docs"])

    print(f"\n=== Cross-doc 评测（{n} 题，top_k={TOP_K}，服务配置） ===")
    print(f"  Doc Recall@k    : {recall:.4f}  ({total_hit}/{total_gold} docs)")
    print(f"  Coverage@k      : {coverage:.2f} 个 gold doc/题")
    print(f"  Doc MRR@k       : {mrr:.4f}")
    print(f"  Full-Coverage   : {full}/{n} = {full / n:.3f}")
    print(f"  AtLeast1 命中率  : {atleast1}/{n} = {atleast1 / n:.3f}")

    print("\n=== 逐题明细 ===")
    for r in records:
        icon = "✅" if set(r["hit_docs"]) == set(r["gold_docs"]) else ("🟡" if r["hit_docs"] else "❌")
        print(f"  {icon} {r['id']}  {r['question'][:38]}")
        print(f"      gold={r['gold_docs']}")
        if r["missed_docs"]:
            print(f"      漏: {r['missed_docs']}")

    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "config": {"top_k": TOP_K, "rewrite": False, "reranker": False, "max_per_doc": 2},
        "metrics": {
            "doc_recall_at_k": round(recall, 4),
            "coverage_at_k": round(coverage, 4),
            "doc_mrr_at_k": round(mrr, 4),
            "full_coverage_rate": round(full / n, 4),
            "at_least_one_rate": round(atleast1 / n, 4),
        },
        "records": records,
    }
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cross_doc_eval_{report['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告: {path}  (耗时 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

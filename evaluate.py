"""
一键评估：python evaluate.py
跑完整个评测管线（系统检索评测 → bad case 诊断 → 历史记录）
等价于 make evaluate

用法:
    python evaluate.py              # 完整评测（重建知识库）
    python evaluate.py --quick      # 快速模式（仅 chunk=500 + top_k=3/5/8）
    python evaluate.py --ablation   # 含消融实验（带 LLM 的变体对比）
    python evaluate.py --skip-reindex  # 增量（已有知识库不重建，只跑评测）
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SKIP_REINDEX = "--skip-reindex" in sys.argv
QUICK = "--quick" in sys.argv
ABLATION = "--ablation" in sys.argv
OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# ── 1. 系统检索质量评测 ──
print("\n" + "=" * 70)
print("  📊 阶段 1：系统检索质量评测" + (" (跳过重索引)" if SKIP_REINDEX else ""))
print("=" * 70)

from eval.metrics import compute_all_metrics, print_metrics_report  # noqa: E402
from eval.test_questions import get_test_questions, print_question_summary  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

questions = get_test_questions()
print_question_summary(questions)

# Milvus 后端：默认 Milvus Lite（免 Docker）；设 MILVUS_LITE=false 用 Docker Milvus
_use_lite = os.getenv("MILVUS_LITE", "true").lower() == "true"
pipeline = RAGPipeline(
    data_dir="data",
    top_k=10,
    enable_rewrite=False,
    enable_reranker=False,
    milvus_lite=_use_lite,
    milvus_host=os.getenv("MILVUS_HOST", "localhost"),
    milvus_port=os.getenv("MILVUS_PORT", "19530"),
    vector_backend="milvus",
)

count = pipeline.initialize_knowledge_base(force_reindex=True) if not SKIP_REINDEX else pipeline.vector_store.count()

if count == 0:
    print("❌ 知识库为空，退出")
    sys.exit(1)
print(f"\n📚 知识库: {count} chunks\n")

if hasattr(pipeline.embedding_provider, "warmup"):
    pipeline.embedding_provider.warmup()
pipeline.retriever._ensure_bm25_index()

records = []
t0 = time.time()
for q in questions:
    question = q["question"]
    cat = q.get("category", "unknown")
    expected = q.get("expected_doc", "")
    diff = q.get("difficulty", "unknown")

    emb = pipeline.embedding_provider.embed([question], prefix="query: ")[0]
    vec_results = pipeline.vector_store.similarity_search(query_embedding=emb, top_k=20)
    bm25_results = pipeline.retriever._bm25_retrieve(question, top_k=20)

    if bm25_results and vec_results:
        sources = pipeline.retriever._rrf_fusion(vec_results, bm25_results, top_k=10)
    elif vec_results:
        sources = vec_results[:10]
    else:
        sources = []

    vec_score = sources[0].get("_vector_score", 0) if sources else 0
    rrf_score = sources[0]["score"] if sources else 0
    top_score = vec_score or rrf_score

    expected_hit = False
    expected_base = expected.rsplit(".", 1)[0] if expected else ""
    gold_chunk_ids = set(q.get("gold_evidence", {}).get("answer_bearing_chunk_ids", []))
    if sources:
        if gold_chunk_ids:
            # chunk-level Gold：任一 answer-bearing chunk 出现在检索结果中即命中
            expected_hit = any(s["id"] in gold_chunk_ids for s in sources)
        elif expected:
            # 兼容旧标注：document-level 命中
            expected_hit = any(
                expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
                for s in sources
            )

    records.append(
        {
            "question": question,
            "category": cat,
            "difficulty": diff,
            "expected_doc": expected,
            "expected_hit": expected_hit,
            "gold_chunk_ids": sorted(gold_chunk_ids),
            "gold_answerability": q.get("gold_evidence", {}).get("answerability", ""),
            "num_retrieved": len(sources),
            "top_score": round(top_score, 4),
            "time_seconds": round(time.time() - t0, 2),
            "sources": sources,
        }
    )

    status = "✅" if expected_hit else (" - " if not expected else "❌")
    print(f"  {status} [{cat[:4]}/{diff[:4]}] {question[:45]:<45s} score={top_score:.3f}")

elapsed = time.time() - t0
metrics = compute_all_metrics(records)
print_metrics_report(metrics)
print(f"  ⏱  总耗时: {elapsed:.1f}s  ({(elapsed / len(questions)):.1f}s/题)\n")

# ── 2. Bad Case 诊断（可选：诊断脚本已随归档清理移出仓库） ──
print("=" * 70)
print("  🔍 阶段 2：Bad Case 诊断")
print("=" * 70)
bad_cases = []
try:
    from scripts.bad_case_review import find_bad_cases, print_bad_case_report  # noqa: E402

    bad_cases = find_bad_cases(records)
    print_bad_case_report(bad_cases)
except (ImportError, ModuleNotFoundError):
    print("  ⚠️ bad_case 诊断脚本已移出仓库（全文见 git 历史中的 scripts/archive/），跳过 bad case 诊断")

pipeline.close()

# ── 3. 保存结果 ──
system_report = {
    "timestamp": TIMESTAMP,
    "config": {"top_k": 10, "chunk": "300-500", "milvus_lite": True},
    "metrics": metrics,
    "records": records,
    "bad_cases": bad_cases,
}
system_path = str(OUT_DIR / f"system_eval_{TIMESTAMP}.json")
with open(system_path, "w", encoding="utf-8") as f:
    json.dump(system_report, f, ensure_ascii=False, indent=2, default=str)
print(f"\n  📄 报告: {system_path}")

# ── 4. 消融实验（可选） ──
ablation_results = None
if ABLATION:
    print("\n" + "=" * 70)
    print("  🔬 阶段 4：消融实验")
    print("=" * 70)
    from eval.evaluate_retrieval import RetrievalEvaluator, print_ablation_table

    ret_eval = RetrievalEvaluator(data_dir="data")
    ablation_results = ret_eval.run_ablation(questions)
    print_ablation_table(ablation_results)

# ── 5. 历史趋势 ──
history_file = OUT_DIR / "eval_history.jsonl"
entry = {
    "timestamp": TIMESTAMP,
    "mode": "quick" if QUICK else ("ablation" if ABLATION else "full"),
    "hit_rate": metrics["overall"]["hit_rate"],
    "mrr": metrics["overall"]["mrr"],
    "semantic_score": metrics["semantic_score"],
    "refusal_accuracy": metrics["overall"]["refusal_accuracy"],
    "bad_case_count": len(bad_cases),
}
with open(history_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"  📈 历史: {history_file}")

# ── 趋势打印 ──
history = [json.loads(l) for l in open(history_file, encoding="utf-8")]
if len(history) > 1:
    print("\n  📊 历史趋势:")
    print(f"  {'日期':<16} {'模式':<10} {'Hit Rate':>10} {'MRR':>8} {'语义分':>8} {'拒答':>8} {'Bad Case':>9}")
    print(f"  {'-' * 72}")
    for h in history[-5:]:
        print(
            f"  {h['timestamp'][:12]:<16} {h['mode']:<10} {h['hit_rate']:>9.0%} "
            f"{h['mrr']:>7.3f} {h['semantic_score']:>7.4f} {h['refusal_accuracy']:>7.0%} "
            f"{h['bad_case_count']:>8}"
        )

print("\n" + "=" * 70)
print("  ✅ 评测完成")
print("=" * 70)

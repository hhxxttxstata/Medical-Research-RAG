"""
完整 RAG Pipeline 评估（含 LLM 生成 + 拒答逻辑）
用法: PYTHONIOENCODING=utf-8 python eval/run_full_pipeline_eval.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from eval.metrics import compute_all_metrics, print_metrics_report  # noqa: E402
from eval.test_questions import get_test_questions  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)

questions = get_test_questions()
print(f"📋 测试集: {len(questions)} 题")

reranker = CrossEncoderReranker()
reranker._load_model()

pipeline = RAGPipeline(
    top_k=10,
    enable_rewrite=True,
    enable_reranker=True,
    reranker=reranker if reranker.model_ready else None,
    milvus_host="localhost",
    milvus_port="19530",
)
pipeline.retriever._ensure_bm25_index()

records = []
t0_tot = time.time()

for i, q in enumerate(questions):
    t0 = time.time()
    try:
        result = pipeline.query(q["question"], top_k=10)
    except Exception as e:
        print(f"  ❌ [{i + 1}/{len(questions)}] 失败: {e}")
        continue
    elapsed = time.time() - t0

    expected_doc = q.get("expected_doc", "")
    sources = result.get("sources", [])
    # expected_doc 形如 "xxx.md"，而 filename 是去扩展名的 stem——需先去掉扩展名再匹配
    expected_stem = Path(expected_doc).stem if expected_doc else ""
    expected_hit = (
        bool(expected_stem and any(expected_stem in s.get("metadata", {}).get("filename", "") for s in sources))
        if expected_doc
        else None
    )

    cat = q.get("category", "unknown")
    diff = q.get("difficulty", "unknown")
    is_refusal = result.get("is_refusal", False)
    correct_refusal = is_refusal if cat == "out_of_knowledge" else not is_refusal

    # 检查回答中是否包含拒答信号
    answer = result.get("answer", "")
    refusal_phrases = ["无法回答", "不涉及", "不在知识库", "超出", "I cannot", "I don't have", "拒答", "out of domain"]
    is_refusal_in_answer = any(p in answer for p in refusal_phrases)

    records.append(
        {
            "question": q["question"],
            "category": cat,
            "difficulty": diff,
            "expected_doc": expected_doc,
            "expected_hit": expected_hit,
            "num_retrieved": len(sources),
            "top_score": sources[0].get("score", 0) if sources else 0,
            "time_seconds": round(elapsed, 2),
            "is_refusal": is_refusal,
            "correct_refusal": correct_refusal,
            "is_refusal_in_answer": is_refusal_in_answer,
            "answer_preview": answer[:120],
            "sources": sources,
        }
    )

    icon = "✅" if expected_hit else ("  " if expected_doc is None else "❌")
    print(f"  {icon} [{cat[:4]}/{diff[:4]}] {q['question'][:40]:<40s} {elapsed:.1f}s")

metrics = compute_all_metrics(records)
print_metrics_report(metrics)
print(f"\n⏱ 总耗时: {time.time() - t0_tot:.0f}s\n")

report = {
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "config": {"top_k": 10, "rewrite": True, "reranker": True},
    "metrics": metrics,
    "records": records,
}
path = str(OUT_DIR / f"full_pipeline_eval_{report['timestamp']}.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
print(f"📄 报告: {path}")

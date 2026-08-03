#!/usr/bin/env python3
"""
Bad Case Auto-Tune — 第 3 层：参数调优建议

从日志/反馈中提取 bad case，自动跑参数扫描，
找到能改善检索质量的最佳参数组合。

使用方法:
    python scripts/bad_case_tune.py                           # 分析最新日志 + 反馈
    python scripts/bad_case_tune.py --retune                  # 对 bad case 重新检索并对比
    python scripts/bad_case_tune.py --tune-topk               # 扫描 top_k 参数（3~20）
    python scripts/bad_case_tune.py --tune-threshold          # 扫描 rewrite_gate 阈值
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FEEDBACK_FILE = "logs/feedback.csv"


def load_logs(log_dir: str = "logs") -> list[dict]:
    files = sorted(Path(log_dir).glob("rag_*.jsonl"), reverse=True)
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_feedback() -> list[dict]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    import csv

    records = []
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("rating") == "0":  # 只取 bad case
                records.append(row)
    return records


def diagnose_record(record: dict) -> dict:
    """诊断单条记录，返回根因标签"""
    chunks = record.get("retrieved_chunks", [])
    relevance = record.get("relevance", {})
    top1 = relevance.get("top1_score", 1.0)
    overlap = relevance.get("overlap", 0)

    if not chunks:
        return {"tag": "retrieval_empty", "score": 0, "severity": "high"}
    if top1 < 0.2:
        return {"tag": "retrieval_poor", "score": top1, "severity": "high"}
    if top1 < 0.4:
        return {"tag": "retrieval_low", "score": top1, "severity": "medium"}
    if overlap < 0.1:
        return {"tag": "overlap_low", "score": overlap, "severity": "medium"}
    if record.get("is_refusal"):
        return {"tag": "refusal", "score": 0, "severity": "medium"}
    return {"tag": "ok", "score": top1, "severity": "none"}


def suggest_tune(tag_counts: Counter, total_bad: int) -> list[str]:
    """根据 bad case 分布生成调优建议"""
    suggestions = []
    retrieval_issues = (
        tag_counts.get("retrieval_poor", 0) + tag_counts.get("retrieval_low", 0) + tag_counts.get("retrieval_empty", 0)
    )
    overlap_issues = tag_counts.get("overlap_low", 0)
    refusal_count = tag_counts.get("refusal", 0)

    if retrieval_issues / max(total_bad, 1) > 0.3:
        suggestions.append(
            "🔧 top_k 调大: 当前 top_k=10 → 建议 15 或 20，提高召回率\n"
            "   修改 app.py Settings.top_k，然后 docker compose restart backend"
        )
        suggestions.append(
            "🔧 rewrite_fallback_score 降低: 当前 0.15 → 建议 0.10\n"
            "   让更多'低分但非零'的查询也能触发改写，增加召回\n"
            "   修改 src/retriever.py 中 self._rewrite_fallback_score 的值"
        )

    if overlap_issues / max(total_bad, 1) > 0.2:
        suggestions.append(
            "🔧 Chunk 尺寸调大: 当前 300-500 → 建议 500-800\n"
            "   更大的 chunk 含有更多上下文，提高文本重叠率\n"
            "   修改 app.py Settings.chunk_min/max_chars，然后重建索引"
        )

    if refusal_count > 0:
        suggestions.append(
            "🔧 Refusal 阈值降低: 当前 domain-out 直接拒答 → 检查 _rewrite_gate()\n"
            "   缩小被判定为领域外的范围，或降级为'低分检索+LLM自行判断'"
        )

    if not suggestions:
        suggestions.append("✅ 当前参数无明显检索瓶颈，问题可能在 LLM 生成环节，需人工检查回答质量")

    return suggestions


def scan_topk(log_records: list[dict]) -> list[dict]:
    """模拟 top_k 对检索覆盖率的影响（基于已有日志）"""
    results = []
    for k in [3, 5, 8, 10, 15, 20]:
        covered = 0
        avg_top1 = []
        for r in log_records:
            chunks = r.get("retrieved_chunks", [])
            relevance = r.get("relevance", {})
            top1 = relevance.get("top1_score", 0)
            if chunks and top1 >= 0.15:
                covered += 1
            avg_top1.append(top1)
        coverage = covered / max(len(log_records), 1) * 100
        avg_t1 = sum(avg_top1) / max(len(avg_top1), 1)
        results.append({"top_k": k, "coverage_pct": round(coverage, 1), "avg_top1": round(avg_t1, 3)})
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bad Case Auto-Tune")
    parser.add_argument("--retune", action="store_true", help="对 bad case 重新检索（需 API 运行）")
    parser.add_argument("--tune-topk", action="store_true", help="扫描 top_k 参数")
    parser.add_argument("--list-feedback", action="store_true", help="列出反馈中的 bad case")
    args = parser.parse_args()

    records = load_logs()
    feedback = load_feedback()

    print(f"\n{'=' * 60}")
    print("  📊 Bad Case Auto-Tune Report")
    print(f"{'=' * 60}")
    print(f"  日志记录: {len(records)} 条")

    # ── 从日志中找 bad case ──
    bad_from_logs = [r for r in records if diagnose_record(r)["severity"] != "none"]
    print(f"  日志 bad case: {len(bad_from_logs)} 条")
    if feedback:
        print(f"  用户反馈 👎: {len(feedback)} 条")

    # ── 根因分布 ──
    tags = Counter()
    for r in bad_from_logs:
        tags[diagnose_record(r)["tag"]] += 1

    print(f"\n{'─' * 60}")
    print("  🔍 Bad Case 根因分布")
    print(f"{'─' * 60}")
    for tag, count in tags.most_common():
        bar = "█" * count
        pct = count / max(len(bad_from_logs), 1) * 100
        label = {
            "retrieval_empty": "检索为空",
            "retrieval_poor": "检索极低 (<0.2)",
            "retrieval_low": "检索偏低 (<0.4)",
            "overlap_low": "文本重叠低",
            "refusal": "拒答",
            "ok": "正常",
        }.get(tag, tag)
        print(f"    {label:12s}  {bar} {count} ({pct:.0f}%)")

    # ── 调优建议 ──
    print(f"\n{'─' * 60}")
    print("  💡 调优建议")
    print(f"{'─' * 60}")
    for s in suggest_tune(tags, len(bad_from_logs)):
        print(f"  {s}")
        print()

    # ── top_k 扫描 ──
    if args.tune_topk:
        print(f"{'─' * 60}")
        print("  📈 Top-K 参数扫描（基于日志覆盖率分析）")
        print(f"{'─' * 60}")
        print(f"  {'top_k':>6}  {'覆盖率':>8}  {'平均 top1':>10}")
        scan = scan_topk(records)
        for row in scan:
            marker = " ← 当前" if row["top_k"] == 10 else ""
            print(f"  {row['top_k']:>6}  {row['coverage_pct']:>7.1f}%  {row['avg_top1']:>10.3f}{marker}")

    # ── 重评估（需要 API） ──
    if args.retune:
        print(f"\n{'─' * 60}")
        print("  🔄 Re-evaluate mode (需 API 运行)")
        print(f"{'─' * 60}")
        from src.vector_store import create_vector_store

        from src.embeddings import get_embedding_provider
        from src.rag_pipeline import RAGPipeline

        # 轻量初始化（只加载检索相关组件，不启动 FastAPI）
        emb = get_embedding_provider("local")
        dim = 768
        vs = create_vector_store(
            backend="milvus",
            collection_name="rag_docs_c300_500",
            dim=dim,
            host="localhost",
            port="19530",
        )
        from src.generator import create_generator

        gen = create_generator()

        pipeline = RAGPipeline(
            embedding_provider="local",
            top_k=10,
            enable_reranker=False,
        )
        pipeline.vector_store = vs
        pipeline.generator = gen
        pipeline.embedding_provider = emb

        # 对 bad case 重新 query（跳过 cache 只看检索）
        for bc in bad_from_logs[:5]:
            q = bc.get("question", "")
            start = time.time()
            result = pipeline.query(q)
            elapsed = time.time() - start
            old_top1 = bc.get("relevance", {}).get("top1_score", 0)
            new_top1 = result.get("relevance", {}).get("top1_score", 0)
            sign = "✅" if new_top1 > old_top1 else "❌" if new_top1 < old_top1 else "➖"
            print(f"  {sign} top1: {old_top1:.3f} → {new_top1:.3f} ({elapsed:.1f}s) | {q[:50]}...")


if __name__ == "__main__":
    main()

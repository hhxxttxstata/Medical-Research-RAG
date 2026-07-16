#!/usr/bin/env python3
"""
Bad Case Review Tool — 从日志中提取 bad case，自动诊断根因

使用方法:
    python scripts/bad_case_review.py                         # 最新的日志
    python scripts/bad_case_review.py --date 2026-07-13       # 指定日期
    python scripts/bad_case_review.py --threshold 0.3          # 自定义阈值
    python scripts/bad_case_review.py --re-evaluate            # 用当前配置重新跑一遍
"""

import json
import os
import sys
from pathlib import Path

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_log_file(date_str: str | None = None) -> str:
    log_dir = "logs"
    if date_str:
        path = os.path.join(log_dir, f"rag_{date_str}.jsonl")
        if os.path.exists(path):
            return path
        print(f"  ❌ {date_str} 的日志不存在")
        sys.exit(1)

    # 找最新的一天的日志
    files = sorted(Path(log_dir).glob("rag_*.jsonl"), reverse=True)
    if not files:
        print("  ❌ logs/ 目录下没有 rag_*.jsonl 日志文件")
        sys.exit(1)
    return str(files[0])


def diagnose(record: dict) -> list[str]:
    """诊断一条 bad case 的可能根因"""
    findings = []
    chunks = record.get("retrieved_chunks", [])
    relevance = record.get("relevance", {})
    top1 = relevance.get("top1_score", 0)
    _avg = relevance.get("avg_score", 0)
    overlap = relevance.get("overlap", 0)

    if not chunks:
        findings.append("🔴 检索为空 — 可能：文档不在知识库中 / Embedding 不匹配")
    elif top1 < 0.2:
        findings.append(f"🔴 检索质量极低 (top1={top1:.2f}) — 可能：Query Rewriting 改坏了 / 文档切分不合理")
    elif top1 < 0.4:
        findings.append(f"🟡 检索质量偏低 (top1={top1:.2f}) — 建议：增加 top_k 或调整 rewrite_gate 阈值")

    if overlap < 0.1 and chunks:
        findings.append(f"🟡 文本重叠率低 (overlap={overlap:.2f}) — 检索到的 chunks 与问题关键词匹配弱")

    if record.get("is_refusal"):
        findings.append("🟡 被拒答 — 可能：Query Rewriting 误判为领域外 / 检索不到相关知识")

    if record.get("elapsed_seconds", 0) > 15:
        findings.append(f"🟠 响应过慢 ({record['elapsed_seconds']:.1f}s) — 可考虑：减少 max_tokens 或并行检索线程")

    if not findings:
        findings.append("✅ 无明显异常 — 可能是 LLM 生成质量或幻觉问题，需人工判断回答内容")

    return findings


def find_bad_cases(records: list[dict], threshold: float = 0.3) -> list[dict]:
    """从评测记录中筛选 bad case"""
    bad = []
    for r in records:
        score = r.get("top_score", 1)
        if score < threshold or r.get("expected_hit") is False:
            bad.append({"record": r, "score": score})
    return bad


def print_bad_case_report(bad_cases: list[dict]) -> None:
    print(f"\n  ⚠️  共 {len(bad_cases)} 条 bad case\n")
    for i, bc in enumerate(bad_cases[:20]):
        r = bc["record"]
        print(f"  [{i + 1}] {r['question'][:50]:<50s} top_score={bc['score']:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bad Case Review Tool")
    parser.add_argument("--date", type=str, help="日志日期 (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=0.3, help="top1_score 阈值 (默认 0.3)")
    parser.add_argument("--re-evaluate", action="store_true", help="用当前配置重新跑一遍并对比")
    parser.add_argument("--max", type=int, default=20, help="最多输出多少条 (默认 20)")
    args = parser.parse_args()

    log_path = find_log_file(args.date)
    print(f"\n📋 分析日志: {log_path}\n")

    # 读取所有记录
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  共 {len(records)} 条查询记录\n")

    # 筛选 bad case
    bad_cases = []
    for r in records:
        rel = r.get("relevance", {})
        top1 = rel.get("top1_score", 1)
        if top1 < args.threshold or r.get("is_refusal") or r.get("error"):
            bad_cases.append(r)

    print(f"⚠️  检出 {len(bad_cases)} 条 bad case (top1_score < {args.threshold} 或拒答/错误)\n")

    # 输出诊断
    for i, bc in enumerate(bad_cases[: args.max]):
        question = bc.get("question", "")[:80]
        top1 = bc.get("relevance", {}).get("top1_score", 0)
        elapsed = bc.get("elapsed_seconds", 0)
        print(f"{'=' * 60}")
        print(f"  [{i + 1}] {question}")
        print(f"      top1={top1:.3f}  | 耗时={elapsed:.1f}s  | 拒答={'是' if bc.get('is_refusal') else '否'}")
        for finding in diagnose(bc):
            print(f"      {finding}")
        print()

    if len(bad_cases) > args.max:
        print(f"  ... 还有 {len(bad_cases) - args.max} 条未显示\n")

    # 统计摘要
    print(f"\n{'=' * 60}")
    print("📊 汇总")
    print(f"{'=' * 60}")
    avg_score_all = sum(r.get("relevance", {}).get("avg_score", 0) for r in records if r.get("relevance")) / max(
        len(records), 1
    )
    print(f"  所有查询平均检索分: {avg_score_all:.3f}")
    print(f"  Bad case 占比: {len(bad_cases)}/{len(records)} ({len(bad_cases) / max(len(records), 1) * 100:.1f}%)")
    print(f"  拒答率: {sum(1 for r in records if r.get('is_refusal')) / max(len(records), 1) * 100:.1f}%")
    print(f"  错误率: {sum(1 for r in records if r.get('error')) / max(len(records), 1) * 100:.1f}%")

    if args.re_evaluate:
        print("\n  Re-evaluate 功能需要 import pipeline，暂未实现。做法：")
        print("  对每条 bad case 的 question 调 pipeline.query()，对比新老结果")


if __name__ == "__main__":
    main()

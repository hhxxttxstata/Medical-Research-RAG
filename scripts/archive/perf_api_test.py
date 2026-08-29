"""
前端性能测试 — 通过 /chat API 端到端延迟测量

用法:
    python scripts/perf_api_test.py [--host http://localhost:8001] [--repeat 2]

测量:
  1. 每道题端到端延迟（检索 + 生成，即前端感知的响应时间）
  2. 按类别汇总（exact_match / cross_doc / out_of_knowledge）
  3. 回答质量信号（是否拒答、回答长度、引用来源数）

产出: eval_results/perf_api_<timestamp>.json
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# 代表性抽样：每类 4 题（easy/medium/hard 均匀），OOD 2 题
SAMPLE_PER_CATEGORY = {"exact_match": 4, "cross_doc": 4, "out_of_knowledge": 2}


def load_samples() -> list[dict]:
    from eval.test_questions import get_test_questions

    qs = get_test_questions()
    random.seed(20260814)  # 可复现抽样
    picked = []
    for cat, n in SAMPLE_PER_CATEGORY.items():
        pool = [q for q in qs if q["category"] == cat]
        picked.extend(random.sample(pool, min(n, len(pool))))
    return picked


def chat_once(host: str, question: str, timeout: int = 120) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(f"{host}/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = round(time.time() - t0, 2)
    return {"api_elapsed_s": elapsed, **data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8001")
    parser.add_argument("--repeat", type=int, default=2, help="每道题重复次数（第 2 次起可能命中回答缓存）")
    args = parser.parse_args()

    samples = load_samples()
    print(f"共 {len(samples)} 道题 × {args.repeat} 次 = {len(samples) * args.repeat} 次调用")
    print("=" * 70)

    records = []
    for q in samples:
        cat, question = q["category"], q["question"]
        for i in range(args.repeat):
            tag = "冷启动(首次)" if i == 0 and cat == samples[0]["category"] else ("缓存?" if i > 0 else "常规")
            r = chat_once(args.host, question)
            elapsed = r["api_elapsed_s"]
            answer = r.get("answer", "")
            is_refusal = r.get("is_refusal", False)
            n_src = len(r.get("sources", []))
            print(
                f"  [{cat[:4]}/{tag}] {elapsed:6.2f}s 拒答={is_refusal} 来源={n_src} 回答={len(answer)}字 | {question[:35]}"
            )
            records.append(
                {
                    "category": cat,
                    "question": question,
                    "repeat": i,
                    "elapsed_s": elapsed,
                    "is_refusal": is_refusal,
                    "answer_len": len(answer),
                    "num_sources": n_src,
                }
            )

    # 汇总
    print("\n" + "=" * 70)
    print("汇总（端到端延迟 = 前端感知响应时间）")
    by_cat: dict[str, list[float]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["elapsed_s"])
    all_times = [r["elapsed_s"] for r in records]
    for cat, times in by_cat.items():
        avg = sum(times) / len(times)
        best, worst = min(times), max(times)
        print(f"  {cat:<18} n={len(times):2d}  avg={avg:6.2f}s  min={best:6.2f}s  max={worst:6.2f}s")
    avg_all = sum(all_times) / len(all_times)
    refusals = sum(1 for r in records if r["is_refusal"])
    print(f"  {'TOTAL':<18} n={len(all_times):2d}  avg={avg_all:6.2f}s  （拒答 {refusals}/{len(all_times)} 题）")

    report = {
        "timestamp": TIMESTAMP,
        "host": args.host,
        "sample_size": len(samples),
        "repeat": args.repeat,
        "summary": {
            "total_avg_s": round(avg_all, 2),
            "by_category": {c: {"n": len(t), "avg_s": round(sum(t) / len(t), 2)} for c, t in by_cat.items()},
            "refusal_count": refusals,
        },
        "records": records,
    }
    out = OUT_DIR / f"perf_api_{TIMESTAMP}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 报告: {out}")


if __name__ == "__main__":
    main()

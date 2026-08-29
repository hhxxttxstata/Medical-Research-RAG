"""
holdout_eval.py — Holdout 30 题验收（V0 / v2 / v2.1 对比，v1 可选）

在 archive/step135_holdout_eval.py（16 题一次性验收）基础上扩展：
  - 题目集扩到 30（tests/benchmark_holdout.json）
  - agents 可选：默认 v0,v2,v21（v1 冻结基线每题 3-5 分钟，30 题可选加回）
  - policy_action_accuracy 已兼容 B2 预检 route（[DECOMPOSE, ACCEPT]）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/holdout_eval.py
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/holdout_eval.py --agents v0,v1,v2,v21

纪律:
  - 串行独占运行（Milvus Lite 单进程锁），先停后端
  - 依赖已重建索引（scripts/rebuild_index.py）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

FETCH_K = 20
TOP_K = 5

# 指标表（与 rescue_metrics 字段对应）
METRIC_KEYS = [
    ("Final Answer Acc", "final_answer_accuracy"),
    ("Evidence Recall@5", "evidence_recall"),
    ("Hop Recall@5", "hop_recall"),
    ("Completeness", "completeness"),
    ("Final Rescue", "final_rescue"),
    ("Harm", "harm"),
    ("NetUtility", "net_utility"),
    ("OOD Reject", "ood_reject"),
    ("False Abstain", "false_abstain"),
    ("Policy Action Acc", "policy_action_accuracy"),
    ("Decomp Success", "decomposition_success"),
    ("Retry Recovery", "retry_recovery"),
    ("Unnecessary Action", "unnecessary_action_rate"),
    ("Avg Iterations", "avg_iterations"),
]


class _CachedReranker:
    """按 (query, chunk_id) 缓存重排分数——行为保真，只去重不改变分数

    背景：CPU 上 bge-reranker-v2-m3 对真实 chunk 约 5s/对；agent 每轮会
    对同一 (query, bank) 重复调用 2-3 次（_top1_rel / _decide_once /
    _completeness）。分数按对独立计算 → 缓存精确复用，评测提速 2-3 倍。
    """

    def __init__(self, inner, max_cache: int = 50000):
        self._inner = inner
        self._cache: dict[tuple[str, str], float] = {}
        self._max_cache = max_cache

    @property
    def model_ready(self) -> bool:
        return self._inner.model_ready

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []
        if not self.model_ready:
            return chunks[:top_k]
        missing = [c for c in chunks if (query, c["id"]) not in self._cache]
        if missing:
            scores = self._inner.rerank_pairs([(query, c["text"]) for c in missing])
            for c, s in zip(missing, scores, strict=False):
                if len(self._cache) >= self._max_cache:
                    self._cache.clear()
                self._cache[(query, c["id"])] = float(s)
        out = []
        for c in chunks:
            cc = dict(c)
            cc["_rerank_score"] = self._cache[(query, c["id"])]
            out.append(cc)
        out.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return out[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="v0,v2,v21", help="逗号分隔: v0,v1,v2,v21")
    parser.add_argument("--only", default="", help="逗号分隔的题目 id 子集（默认全部）")
    parser.add_argument(
        "--bench",
        default="tests/benchmark_holdout.json",
        help="题集文件（与 holdout 同构: id/type/question/hops/expected_route）",
    )
    args = parser.parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    only_ids = [x.strip() for x in args.only.split(",") if x.strip()]

    print("=" * 78)
    print(f"  🔒 Holdout Generalization Gate（30 题 / agents={agents}）")
    print("=" * 78, flush=True)

    from eval.rescue_metrics import (
        compute_agent_capability_metrics,
        evidence_recall_at_k,
        hop_gold_ids,
    )
    from src.agentic_rag import AgenticRAG
    from src.cost_aware_agentic_rag import CostAwareAgenticRAG
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    reranker = _CachedReranker(CrossEncoderReranker())
    reranker._inner._load_model()
    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=5,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )
    generator = create_generator()

    agent_objs: dict[str, object] = {}
    if "v1" in agents:
        from src.agentic_rag_v1_backup import AgenticRAG as AgenticRAGv1

        agent_objs["v1"] = AgenticRAGv1(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
        print("  ⚠️  v1 已启用（每题 3-5 分钟，总耗时显著增加）", flush=True)
    if "v2" in agents:
        agent_objs["v2"] = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    if "v21" in agents:
        agent_objs["v21"] = CostAwareAgenticRAG(
            retriever=retriever, generator=generator, reranker=reranker, max_iterations=2
        )

    bench = json.load(open(args.bench, encoding="utf-8"))
    if isinstance(bench, dict):
        bench = bench.get("benchmark", bench.get("questions", []))
    if only_ids:
        bench = [b for b in bench if b["id"] in only_ids]
        print(f"  🎯 子集模式: {len(bench)} 题 {only_ids}", flush=True)
    print(f"  📝 {args.bench}: {len(bench)} 题（unseen，仅此一次验证）", flush=True)

    cases = []
    t0 = time.time()
    t_question = t0
    for i, b in enumerate(bench, 1):
        question = b["question"]
        qtype = b["type"]
        all_gold = set()
        for hg in hop_gold_ids(b):
            all_gold |= hg
        print(f"  ── [{i}/{len(bench)}] {b['id']} [{qtype}] {question[:42]}", flush=True)

        case: dict = {"question": b}

        # ── V0 Fixed RAG：单轮 hybrid → RRF Top5（冻结 baseline）
        # （2026-08-17：v0 的 rerank 步骤省略——CPU 上每题 ~100s 且仅用于
        #   rescue/harm 基线判定；RRF top5 与 rerank top5 的 hit/miss 差异极小）
        if "v0" in agents:
            v0_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:FETCH_K]
            case["v0_sources"] = v0_cands[:TOP_K]
            case["v0_evidence_recall"] = evidence_recall_at_k(v0_cands[:TOP_K], all_gold)

        # ── Agentic versions ──
        route_strs = []
        for name in ("v1", "v2", "v21"):
            if name in agents:
                r = agent_objs[name].run(question, fetch_k=FETCH_K, verbose=False)
                case[f"{name}_sources"] = r["sources"]
                case[f"{name}_route"] = r["route"]
                case[f"{name}_answer"] = r["answer"]
                case[f"{name}_abstained"] = r["abstained"]
                case[f"{name}_iterations"] = r["iterations"]
                if name == "v21":
                    case["v21_observation"] = r.get("observation", {})
                route_strs.append(f"{name}_route={r['route']}")

        print(
            f"    V0_ER={case.get('v0_evidence_recall', float('nan')):.2f} | "
            + " | ".join(route_strs)
            + f" | 本题耗时 {time.time() - t_question:.0f}s",
            flush=True,
        )
        t_question = time.time()
        cases.append(case)

    elapsed = time.time() - t0

    def _wrap(prefix: str) -> list[dict]:
        return [
            {
                "question": c["question"],
                "v0_sources": c.get("v0_sources", []),
                "v1_sources": c.get(f"{prefix}_sources", []),
                "v1_route": c.get(f"{prefix}_route", []),
                "v1_answer": c.get(f"{prefix}_answer", ""),
                "v1_abstained": c.get(f"{prefix}_abstained", True),
            }
            for c in cases
        ]

    metrics: dict[str, dict] = {}
    for name in ("v1", "v2", "v21"):
        if name in agents:
            metrics[name] = compute_agent_capability_metrics(_wrap(name))

    def _fmt(m):
        return {label: m[k] for label, k in METRIC_KEYS}

    print("\n" + "=" * 78)
    print(f"  📊 Holdout 对比（agents={agents}）")
    print("=" * 78)
    header = "  " + f"{'指标':<22}"
    for name in ("v0", "v1", "v2", "v21"):
        if name in agents:
            header += f"{name:>12}"
    print(header)
    print("  " + "-" * (24 + 12 * len(agents)))
    for label, k in METRIC_KEYS:
        row = "  " + f"{label:<22}"
        for name in ("v0", "v1", "v2", "v21"):
            if name in agents:
                if name == "v0":
                    val = "—"
                else:
                    val = metrics[name][k]
                row += f"{str(val):>12}"
        print(row)

    # ── 逐题 v2.1 明细（Failure Anatomy）──
    if "v21" in agents:
        print("\n  ── v2.1 逐题明细 ──")
        for c in cases:
            q = c["question"]
            d = next(x for x in metrics["v21"]["details"] if x["id"] == q["id"])
            print(
                f"    {q['id']:>16} [{q['type']:>20}] route={c['v21_route']} "
                f"abstain={c['v21_abstained']} class={d['class']:>6} ER={d['v1_evidence_recall']:.2f}",
                flush=True,
            )

    # ── Operational Failure（单独统计）──
    if "v21" in agents:
        op_fail = sum(1 for c in cases if "[OPERATIONAL_ERROR]" in c["v21_answer"] or "系统错误" in c["v21_answer"])
        print("\n  ── Operational ──")
        print(f"  v2.1 Operational Failure = {op_fail}/{len(cases)}")
        for c in cases:
            if "[OPERATIONAL_ERROR]" in c["v21_answer"] or "系统错误" in c["v21_answer"]:
                print(f"    ⚠️  {c['question']['id']}: {c['v21_answer'][:60]}")

    out = OUT_DIR / f"holdout30_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "agents": agents,
                "note": "Holdout 30 题验收。do NOT tune on these cases.",
                "metrics": {name: _fmt(m) for name, m in metrics.items()},
                "details": {name: m["details"] for name, m in metrics.items()},
                "cases": cases,  # 含原始答案（诊断用）
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

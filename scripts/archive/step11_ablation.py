"""
Step 11: Agentic Ablation — Fixed Hybrid vs Agentic v1 vs 组件移除

设计（考虑评测成本）：
  1. 81 题（有 gold 的 40 题精确对比）：
       V0 Fixed Hybrid vs Agentic v1（FinalHit / OOD Reject / False Abstain）
  2. 9 题 Policy Probe Set（route 行为差异）：
       6 变体 × 预期 route 对比 → Route Accuracy / 组件边际贡献

变体：
  V0 Fixed Hybrid     : 单轮检索 → rerank → Top5 → 生成（无 agent 层）
  v1_full             : Policy Node + 循环（完整）
  v1 − Grader         : 禁用 LLM grader（仅 reranker signal + 规则）
  v1 − Retry          : max_iterations=1（无 RETRIEVE 循环）
  v1 − Decompose      : DECOMPOSE 降级为 RETRIEVE
  v1 − Abstain        : 禁 ABSTAIN（证据不足也硬答）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step11_ablation.py

产出: eval_results/step11_ablation_<timestamp>.json
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

FETCH_K = 20
TOP_K = 5


def build_variants(retriever, generator, reranker):
    """构造 6 个变体的 run 函数（输入 question → result dict）"""
    from src.agentic_rag import AgenticRAG

    variants = {}

    # V0: Fixed Hybrid（无 agent 层，单轮）
    def v0_run(question):
        cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:FETCH_K]
        ev = reranker.rerank(question, list(cands), TOP_K)
        return {
            "route": ["RETRIEVE", "ACCEPT"],
            "sources": ev,
            "abstained": False,
            "iterations": 1,
            "state": None,
            "answer": "",
        }

    variants["V0_fixed"] = v0_run

    # Agentic v1 完整
    agent = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    variants["v1_full"] = lambda q, a=agent: a.run(q, fetch_k=FETCH_K, verbose=False)

    # v1 − Grader：禁用 LLM grader
    ag = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    ag.evidence_grade = lambda q, chunks: (
        {"decision": "insufficient", "reason": "grader 禁用，仅用 signal", "evidence_score": 0.0, "mode": "rule"}
        if chunks
        else {"decision": "insufficient", "reason": "无候选", "evidence_score": 0.0, "mode": "rule"}
    )
    variants["v1_no_grader"] = lambda q, a=ag: a.run(q, fetch_k=FETCH_K, verbose=False)

    # v1 − Retry：max_iterations=1
    ar = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=1)
    variants["v1_no_retry"] = lambda q, a=ar: a.run(q, fetch_k=FETCH_K, verbose=False)

    # v1 − Decompose：DECOMPOSE 降级为 RETRIEVE
    ad = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    orig_p = ad.policy
    ad.policy = lambda question, state, grade, _o=orig_p: (
        lambda s, a, m: (s, "RETRIEVE" if a == "DECOMPOSE" else a, m)
    )(*_o(question, state, grade))
    variants["v1_no_decomp"] = lambda q, a=ad: a.run(q, fetch_k=FETCH_K, verbose=False)

    # v1 − Abstain：禁 ABSTAIN（硬答）
    aa = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)
    orig_p2 = aa.policy
    aa.policy = lambda question, state, grade, _o=orig_p2: (lambda s, a, m: (s, "ACCEPT" if a == "ABSTAIN" else a, m))(
        *_o(question, state, grade)
    )
    variants["v1_no_abstain"] = lambda q, a=aa: a.run(q, fetch_k=FETCH_K, verbose=False)

    return variants


def main():
    print("=" * 70)
    print("  🔬 Step 11: Agentic Ablation")
    print("=" * 70, flush=True)

    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.lucene_bm25 import LuceneBM25Index
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    reranker = CrossEncoderReranker()
    reranker._load_model()
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
    variants = build_variants(retriever, generator, reranker)

    questions = json.load(open("tests/test_questions.json", encoding="utf-8"))
    probes = json.load(open("tests/policy_probes.json", encoding="utf-8"))["probes"]

    from eval.rescue_metrics import gold_chunk_ids, hit_at_k

    # ══════════ Part 1: 81 题 — V0 vs v1_full（有 gold 的精确对比）══════════
    print("\n" + "─" * 70)
    print("  Part 1: 81 题 V0 vs v1_full（FinalHit / OOD / 误拒）")
    print("─" * 70, flush=True)
    part1 = {}
    for vname in ("V0_fixed", "v1_full"):
        stats = {
            "hit": 0,
            "n": 0,
            "abstain": 0,
            "ood_ok": 0,
            "ood_total": 0,
            "false_abs": 0,
            "ans_total": 0,
            "iter_sum": 0,
            "route_counts": {},
        }
        for q in questions:
            r = variants[vname](q["question"])
            gold = gold_chunk_ids(q)
            hit = hit_at_k(r["sources"], gold, q.get("expected_doc", ""), k=TOP_K)
            is_ood = q.get("category") == "out_of_knowledge"
            stats["hit"] += hit
            stats["n"] += 1
            stats["abstain"] += r["abstained"]
            stats["iter_sum"] += r["iterations"]
            rk = "→".join(r["route"])
            stats["route_counts"][rk] = stats["route_counts"].get(rk, 0) + 1
            if is_ood:
                stats["ood_total"] += 1
                stats["ood_ok"] += r["abstained"]
            elif q.get("gold_evidence"):
                stats["ans_total"] += 1
                stats["false_abs"] += r["abstained"]
        n = stats["n"]
        part1[vname] = {
            "hit": stats["hit"],
            "hit_rate": round(stats["hit"] / n, 3),
            "abstain": stats["abstain"],
            "ood_reject": f"{stats['ood_ok']}/{stats['ood_total']}",
            "false_abstain": f"{stats['false_abs']}/{stats['ans_total']}",
            "avg_iterations": round(stats["iter_sum"] / n, 2),
            "route_counts": stats["route_counts"],
        }
        print(
            f"  {vname:<10} Hit={stats['hit']}/{n} OOD拒答={stats['ood_ok']}/{stats['ood_total']} "
            f"误拒={stats['false_abs']}/{stats['ans_total']} avg_iter={part1[vname]['avg_iterations']}",
            flush=True,
        )

    # ══════════ Part 2: 9 题 Probe Set — 6 变体 route 行为 ══════════
    print("\n" + "─" * 70)
    print("  Part 2: 9 题 Probe Set — 6 变体 Route Accuracy")
    print("─" * 70, flush=True)
    part2 = {}
    for vname, vrun in variants.items():
        stats = {"route_ok": 0, "n": 0, "decompose": 0, "retrieve_loop": 0, "abstain": 0, "hit": 0, "by_cat": {}}
        for p in probes:
            r = vrun(p["question"])
            expected = p["expected_route"]
            # 宽松匹配：终局动作一致（retrieve/decompose 类要求循环内动作出现）
            if p["category"] in ("accept", "abstain"):
                ok = r["route"][-1] == expected[-1]
            else:
                ok = r["route"][-1] == expected[-1] and any(a in r["route"][1:-1] for a in ("RETRIEVE", "DECOMPOSE"))
            stats["n"] += 1
            stats["route_ok"] += ok
            stats["decompose"] += "DECOMPOSE" in r["route"]
            stats["retrieve_loop"] += any(a == "RETRIEVE" for a in r["route"][1:-1])
            stats["abstain"] += r["abstained"]
            gold = set(p["gold_chunk_ids"])
            stats["hit"] += bool({c["id"] for c in r["sources"]} & gold) and bool(gold)
            stats["by_cat"].setdefault(p["category"], [0, 0])[0] += ok
            stats["by_cat"].setdefault(p["category"], [0, 0])[1] += 1
        part2[vname] = {
            "route_accuracy": f"{stats['route_ok']}/{stats['n']}",
            "decompose_triggered": stats["decompose"],
            "retrieve_loop_triggered": stats["retrieve_loop"],
            "abstain": stats["abstain"],
            "gold_hit": stats["hit"],
            "by_category": {k: f"{v[0]}/{v[1]}" for k, v in stats["by_cat"].items()},
        }
        print(
            f"  {vname:<14} RouteAcc={stats['route_ok']}/{stats['n']} DECOMPOSE={stats['decompose']} "
            f"RetryLoop={stats['retrieve_loop']} Abstain={stats['abstain']} gold_hit={stats['hit']}",
            flush=True,
        )

    # ── 汇总表 ──
    print("\n" + "=" * 70)
    print("  📊 Ablation 汇总")
    print("=" * 70)
    print(f"  {'Variant':<14} {'Hit@5(81题)':>10} {'OOD拒答':>8} {'误拒':>8} {'ProbeRoute':>10} {'ProbeDECOMP':>11}")
    for vname in part1:
        p1, p2 = part1[vname], part2[vname]
        print(
            f"  {vname:<14} {p1['hit']:>4}/{p1['hit_rate']:<6} {p1['ood_reject']:>8} "
            f"{p1['false_abstain']:>8} {p2['route_accuracy']:>10} {p2['decompose_triggered']:>11}"
        )
    # 只跑过 probe 的变体（组件移除）单独列
    print(f"  {'─' * 14} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 11}")
    for vname in part2:
        if vname not in part1:
            p2 = part2[vname]
            print(
                f"  {vname:<14} {'—':>10} {'—':>8} {'—':>8} {p2['route_accuracy']:>10} {p2['decompose_triggered']:>11}"
            )

    out = OUT_DIR / f"step11_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"part1_81q": part1, "part2_probe": part2, "timestamp": TIMESTAMP}, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

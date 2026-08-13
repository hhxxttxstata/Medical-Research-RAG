"""
Step 10.5: Agent Policy 资格审计

目标：回答 next_step.md 的核心问题——
  1. RETRIEVE / DECOMPOSE 是否在真实评测中被触发过？
  2. decide() 是"程序规则完整规定"还是"模型根据 state 输出 action"？
  3. 每题的 grade 细节（mode: llm/rule、每轮 decision、gold 名次）

审计结果 → 决定冻结为 Adaptive RAG v1 还是 Agentic RAG v1。

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step105_policy_audit.py

产出: eval_results/step105_policy_audit_<timestamp>.json
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

from src.agentic_rag import AgenticRAG  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

FETCH_K = 20


def main():
    print("=" * 70)
    print("  🔬 Step 10.5: Agent Policy 资格审计")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)
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
    agent = AgenticRAG(retriever=retriever, generator=None, reranker=reranker, max_iterations=2)

    questions = json.load(open("tests/test_questions.json", encoding="utf-8"))

    cases = []
    stats = {
        "route_counts": Counter(),
        "grade_modes": Counter(),  # 每轮 grade 的 mode 分布
        "decision_counts": Counter(),  # 每轮 grade 的 decision 分布
        "gold_in_initial_candidates": 0,  # gold 在初始检索候选里
        "gold_in_final_candidates": 0,  # gold 在最终累积候选里
        "gold_in_final_evidence": 0,  # gold 在 final_evidence(top5) 里
        "retrieve_triggered": 0,  # 循环内 RETRIEVE 触发次数
        "decompose_triggered": 0,  # DECOMPOSE 触发次数
        "final_retry_triggered": 0,  # 终局 final-retry 次数
        "top1_gold_rank": [],  # gold 在候选池中的最佳名次
    }

    t0 = time.time()
    for i, q in enumerate(questions, 1):
        question = q["question"]
        gold_ids = set(q.get("gold_evidence", {}).get("answer_bearing_chunk_ids", []))
        cat = q.get("category", "")

        print(f"  ── [{i}/{len(questions)}] {q['id']} {question[:40]}", flush=True)

        # ── 逐轮手动跑循环，记录 grade 细节 ──
        state = agent._make_state(question) if hasattr(agent, "_make_state") else None
        # 直接用 run 但开启 verbose；同时手工记录 grade 日志
        grade_log = []
        route = []
        seen_retrieve_loop, seen_decompose = False, False
        final_retry = False

        # 手动复刻 run() 循环以便记录细节（避免依赖 run 的内部输出）
        from src.agentic_rag import AgentState

        state = AgentState(original_query=question)
        route.append("RETRIEVE")
        initial = agent.hybrid_search(question, fetch_k=FETCH_K, note="initial")
        state.retrieval_history.append({"query": question, "sources": initial, "iteration": 0, "reason": "initial"})
        agent._dedup_accumulate(state, initial)
        state.iteration += 1

        grade = agent.evidence_grade(question, state.candidates)
        state.evidence_score = grade["evidence_score"]
        grade_log.append(
            {
                "round": state.iteration,
                "decision": grade["decision"],
                "score": grade["evidence_score"],
                "mode": grade.get("mode", "?"),
            }
        )
        _, decision, _ = agent.policy(question, state, grade)

        while decision in ("RETRIEVE", "DECOMPOSE") and state.iteration < agent.max_iterations:
            route.append(decision)  # 记录循环内动作
            if decision == "DECOMPOSE":
                seen_decompose = True
                subs = agent.decompose(question)
                if subs:
                    for sub in subs:
                        r = agent.hybrid_search(sub, fetch_k=FETCH_K, note=f"decompose:{sub}")
                        state.retrieval_history.append(
                            {"query": sub, "sources": r, "iteration": state.iteration, "reason": "decompose"}
                        )
                        agent._dedup_accumulate(state, r)
                else:
                    r = agent.hybrid_search(question, fetch_k=FETCH_K, note="retry-decompose-fail")
                    state.retrieval_history.append(
                        {"query": question, "sources": r, "iteration": state.iteration, "reason": "retry"}
                    )
                    agent._dedup_accumulate(state, r)
            else:  # RETRIEVE
                seen_retrieve_loop = True
                nq = agent._build_retrieval_variant(state, question)
                r = agent.hybrid_search(nq, fetch_k=FETCH_K, note=f"retrieve:{nq}")
                state.retrieval_history.append(
                    {"query": nq, "sources": r, "iteration": state.iteration, "reason": "retrieve"}
                )
                agent._dedup_accumulate(state, r)
            state.iteration += 1
            grade = agent.evidence_grade(question, state.candidates)
            state.evidence_score = grade["evidence_score"]
            grade_log.append(
                {
                    "round": state.iteration,
                    "decision": grade["decision"],
                    "score": grade["evidence_score"],
                    "mode": grade.get("mode", "?"),
                }
            )
            _, decision, _ = agent.policy(question, state, grade)

        if decision != "ACCEPT" and decision != "ABSTAIN":
            # 终局处理（可能 final-retry）
            if state.iteration < agent.max_iterations:
                final_retry = True
                nq = agent._build_retrieval_variant(state, question, deeper=True)
                r = agent.hybrid_search(nq, fetch_k=FETCH_K * 2, note="final-retry")
                state.retrieval_history.append(
                    {"query": nq, "sources": r, "iteration": state.iteration, "reason": "final-retry"}
                )
                agent._dedup_accumulate(state, r)
                state.iteration += 1
                grade = agent.evidence_grade(question, state.candidates)
                state.evidence_score = grade["evidence_score"]
                grade_log.append(
                    {
                        "round": state.iteration,
                        "decision": grade["decision"],
                        "score": grade["evidence_score"],
                        "mode": grade.get("mode", "?"),
                    }
                )
                decision = agent.policy(question, state, grade)[1]
            if decision not in ("ACCEPT", "ABSTAIN"):
                decision = "ABSTAIN"

        # ── 统计 ──
        cand_ids = [c["id"] for c in state.candidates]
        gold_ranks = [i for i, cid in enumerate(cand_ids) if cid in gold_ids]
        best_rank = min(gold_ranks) if gold_ranks else -1

        init_ids = {c["id"] for c in initial}
        stats["gold_in_initial_candidates"] += bool(init_ids & gold_ids) and bool(gold_ids)
        stats["gold_in_final_candidates"] += bool(gold_ranks) and bool(gold_ids)
        final_ev = agent._select_final_evidence(question, state.candidates, 5)
        stats["gold_in_final_evidence"] += bool({c["id"] for c in final_ev} & gold_ids) and bool(gold_ids)
        stats["retrieve_triggered"] += seen_retrieve_loop
        stats["decompose_triggered"] += seen_decompose
        stats["final_retry_triggered"] += final_retry
        if best_rank >= 0:
            stats["top1_gold_rank"].append(best_rank)

        route.append(decision)
        stats["route_counts"]["→".join(route)] += 1
        # 终局 final-retry 是否真实发生
        if final_retry and grade["decision"] == "sufficient":
            stats["final_retry_triggered"] += 1
        for g in grade_log:
            stats["grade_modes"][g["mode"]] += 1
            stats["decision_counts"][f"{g['decision']}({g['mode']})"] += 1

        cases.append(
            {
                "id": q["id"],
                "category": cat,
                "question": question,
                "route": route,
                "grade_log": grade_log,
                "best_gold_rank": best_rank,
                "num_candidates": len(cand_ids),
                "evidence_score": state.evidence_score,
            }
        )

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("  📊 Policy 资格审计结果")
    print("=" * 70)
    print(f"  N = {len(questions)} 题")
    print(f"  route 分布        = {dict(stats['route_counts'])}")
    print(f"  循环内 RETRIEVE 触发 = {stats['retrieve_triggered']} 题")
    print(f"  DECOMPOSE 触发     = {stats['decompose_triggered']} 题")
    print(f"  final-retry 触发   = {stats['final_retry_triggered']} 题")
    print(f"  grade mode 分布    = {dict(stats['grade_modes'])}")
    print(f"  grade decision    = {dict(stats['decision_counts'])}")
    print(f"  gold@initial candidates = {stats['gold_in_initial_candidates']}")
    print(f"  gold@final candidates   = {stats['gold_in_final_candidates']}")
    print(f"  gold@final evidence(top5)= {stats['gold_in_final_evidence']}")
    if stats["top1_gold_rank"]:
        print(
            f"  gold 最佳名次中位数 = {sorted(stats['top1_gold_rank'])[len(stats['top1_gold_rank']) // 2]} "
            f"(n={len(stats['top1_gold_rank'])})"
        )

    out = OUT_DIR / f"step105_policy_audit_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
                "cases": cases,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

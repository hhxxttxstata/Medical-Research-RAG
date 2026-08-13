"""
Step 10.5: Policy Probe Set 评测

对 tests/policy_probes.json 的 4 类 probe（accept / retrieve / decompose / abstain）
逐题跑 Agentic RAG 完整循环，记录：
  - 实际 route（含循环内动作）
  - 每轮 grade（mode: llm/rule, decision, score）
  - gold 是否进最终 evidence

指标（next_step.md 定义）：
  - Route Accuracy: 实际 route == expected_route（宽松版：只看终局动作）
  - Tool Selection Accuracy: 每个动作是否被正确选择
  - Recovery Rate: RETRIEVE/DECOMPOSE 后最终是否 ACCEPT 且 gold 命中
  - False Abstain: 应回答却拒答
  - Average Iterations / LLM Calls / Retrieval Calls

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step105_probe_eval.py

产出: eval_results/step105_probe_eval_<timestamp>.json
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


def main():
    print("=" * 70)
    print("  🔬 Step 10.5: Policy Probe Set 评测")
    print("=" * 70, flush=True)

    from src.agentic_rag import AgenticRAG, AgentState
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
    agent = AgenticRAG(retriever=retriever, generator=create_generator(), reranker=reranker, max_iterations=2)

    probes = json.load(open("tests/policy_probes.json", encoding="utf-8"))["probes"]

    results = []
    stats = {
        "route_accuracy": 0,
        "n_probes": 0,
        "tool_select_accuracy": 0,
        "n_actions": 0,
        "recovery": 0,
        "n_recoverable": 0,
        "false_abstain": 0,
        "n_answerable": 0,
        "avg_iterations": 0.0,
        "llm_calls": 0,
        "retrieval_calls": 0,
        "by_category": {},
    }

    t0 = time.time()
    for i, p in enumerate(probes, 1):
        question = p["question"]
        gold = set(p["gold_chunk_ids"])
        expected = p["expected_route"]
        cat = p["category"]
        print(f"  ── [{i}/{len(probes)}] {p['id']} [{cat}] {question[:45]}", flush=True)

        # ── 完整循环，逐轮记录 ──
        state = AgentState(original_query=question)
        route = ["RETRIEVE"]
        grade_log = []
        llm_calls = 0
        retrieval_calls = 0

        initial = agent.hybrid_search(question, fetch_k=FETCH_K, note="initial")
        state.retrieval_history.append({"query": question, "sources": initial, "iteration": 0, "reason": "initial"})
        agent._dedup_accumulate(state, initial)
        state.iteration += 1
        retrieval_calls += 1

        grade = agent.evidence_grade(question, state.candidates)
        state.evidence_score = grade["evidence_score"]
        if grade.get("mode") == "llm":
            llm_calls += 1
        grade_log.append({"round": 1, **{k: grade[k] for k in ("decision", "evidence_score", "mode")}})
        _, decision, pmode = agent.policy(question, state, grade)

        while decision in ("RETRIEVE", "DECOMPOSE") and state.iteration < agent.max_iterations:
            route.append(decision)
            if decision == "DECOMPOSE":
                subs = agent.decompose(question)
                if subs:
                    for sub in subs:
                        r = agent.hybrid_search(sub, fetch_k=FETCH_K, note=f"decompose:{sub}")
                        state.retrieval_history.append(
                            {"query": sub, "sources": r, "iteration": state.iteration, "reason": "decompose"}
                        )
                        agent._dedup_accumulate(state, r)
                        retrieval_calls += 1
                else:
                    r = agent.hybrid_search(question, fetch_k=FETCH_K, note="retry-decompose-fail")
                    state.retrieval_history.append(
                        {"query": question, "sources": r, "iteration": state.iteration, "reason": "retry"}
                    )
                    agent._dedup_accumulate(state, r)
                    retrieval_calls += 1
            else:
                nq = agent._build_retrieval_variant(state, question)
                r = agent.hybrid_search(nq, fetch_k=FETCH_K, note=f"retrieve:{nq}")
                state.retrieval_history.append(
                    {"query": nq, "sources": r, "iteration": state.iteration, "reason": "retrieve"}
                )
                agent._dedup_accumulate(state, r)
                retrieval_calls += 1
            state.iteration += 1
            grade = agent.evidence_grade(question, state.candidates)
            state.evidence_score = grade["evidence_score"]
            if grade.get("mode") == "llm":
                llm_calls += 1
            grade_log.append(
                {"round": state.iteration, **{k: grade[k] for k in ("decision", "evidence_score", "mode")}}
            )
            _, decision, pmode = agent.policy(question, state, grade)

        # ── 终局决策（与 agent 语义一致：ABSTAIN 只代表最终拒答，无降级）──
        if decision != "ACCEPT" and decision != "ABSTAIN":
            # RETRIEVE/DECOMPOSE 迭代用尽 → final-retry
            if state.iteration < agent.max_iterations:
                nq = agent._build_retrieval_variant(state, question, deeper=True)
                r = agent.hybrid_search(nq, fetch_k=FETCH_K * 2, note="final-retry")
                state.retrieval_history.append(
                    {"query": nq, "sources": r, "iteration": state.iteration, "reason": "final-retry"}
                )
                agent._dedup_accumulate(state, r)
                state.iteration += 1
                retrieval_calls += 1
                grade = agent.evidence_grade(question, state.candidates)
                state.evidence_score = grade["evidence_score"]
                if grade.get("mode") == "llm":
                    llm_calls += 1
                grade_log.append(
                    {"round": state.iteration, **{k: grade[k] for k in ("decision", "evidence_score", "mode")}}
                )
                decision = agent.policy(question, state, grade)[1]
            if decision not in ("ACCEPT", "ABSTAIN"):
                decision = "ABSTAIN"

        route.append(decision)

        # ── final evidence + gold 命中 ──
        final_ev = agent._select_final_evidence(question, state.candidates, 5)
        hit = bool({c["id"] for c in final_ev} & gold) and bool(gold)
        gold_in_cands = bool({c["id"] for c in state.candidates} & gold) and bool(gold)

        # ── 指标 ──
        # Route Accuracy: 终局动作匹配（宽松） + 含 DECOMPOSE 的路径匹配（严格）
        strict_match = route == expected
        # 宽松：终局动作一致（对 retrieve/decompose 类，要求循环内动作也在）
        if cat in ("accept", "abstain"):
            loose_match = route[-1] == expected[-1]
        else:
            # retrieve/decompose 类要求循环内动作出现
            loop_actions = [a for a in route[1:-1]]
            loose_match = route[-1] == expected[-1] and bool(loop_actions)
        match = strict_match or loose_match

        stats["n_probes"] += 1
        stats["route_accuracy"] += match
        stats["avg_iterations"] += state.iteration
        stats["llm_calls"] += llm_calls
        stats["retrieval_calls"] += retrieval_calls
        stats["by_category"].setdefault(cat, {"n": 0, "route_ok": 0, "hit": 0, "false_abstain": 0})
        stats["by_category"][cat]["n"] += 1
        stats["by_category"][cat]["route_ok"] += match
        stats["by_category"][cat]["hit"] += hit

        # Tool Selection Accuracy: 期望动作序列中的每个动作是否被选择
        for act in expected:
            stats["n_actions"] += 1
            stats["tool_select_accuracy"] += act in route

        # Recovery: 循环内触发了 RETRIEVE/DECOMPOSE 且最终 ACCEPT+hit
        if cat in ("retrieve", "decompose"):
            stats["n_recoverable"] += 1
            if len(route) > 2 and route[-1] == "ACCEPT" and hit:
                stats["recovery"] += 1

        # False Abstain
        if cat in ("accept", "retrieve", "decompose"):
            stats["n_answerable"] += 1
            if route[-1] == "ABSTAIN":
                stats["false_abstain"] += 1

        verdict = "✅" if match else "❌"
        print(
            f"    {verdict} route={route} (期望 {expected}) | grade={[g['decision'][:6] for g in grade_log]} "
            f"| gold_hit={hit} | iters={state.iteration} | llm={llm_calls} retr={retrieval_calls}"
        )

        results.append(
            {
                "id": p["id"],
                "category": cat,
                "question": question,
                "route": route,
                "expected_route": expected,
                "route_match": match,
                "gold_hit": hit,
                "gold_in_candidates": gold_in_cands,
                "grade_log": grade_log,
                "llm_calls": llm_calls,
                "retrieval_calls": retrieval_calls,
                "iterations": state.iteration,
            }
        )

    elapsed = time.time() - t0
    n = stats["n_probes"]
    stats["route_accuracy"] = round(stats["route_accuracy"] / n, 3)
    stats["tool_select_accuracy"] = round(stats["tool_select_accuracy"] / stats["n_actions"], 3)
    stats["recovery"] = f"{stats['recovery']}/{stats['n_recoverable']}"
    stats["false_abstain"] = f"{stats['false_abstain']}/{stats['n_answerable']}"
    stats["avg_iterations"] = round(stats["avg_iterations"] / n, 2)

    print("\n" + "=" * 70)
    print("  📊 Policy Probe Set 结果")
    print("=" * 70)
    print(f"  N = {n} probes")
    print(f"  Route Accuracy      = {stats['route_accuracy']}")
    print(f"  Tool Select Acc     = {stats['tool_select_accuracy']}")
    print(f"  Recovery Rate       = {stats['recovery']}")
    print(f"  False Abstain       = {stats['false_abstain']}")
    print(f"  Avg Iterations      = {stats['avg_iterations']}")
    print(f"  LLM Calls (total)   = {stats['llm_calls']}")
    print(f"  Retrieval Calls     = {stats['retrieval_calls']}")
    for cat, s in stats["by_category"].items():
        print(f"    [{cat}] n={s['n']} route_ok={s['route_ok']} hit={s['hit']}")

    out = OUT_DIR / f"step105_probe_eval_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": results, "elapsed": round(elapsed, 1)}, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

"""
Step 10.5: Counterfactual Policy Test

核心问题：同一个 Query，在 3 种不同 evidence state 下，Policy Node 是否输出不同 action？

  State A (sufficient)  : 候选 = gold chunks（完整 answer-bearing evidence）
                          → 期望 ACCEPT
  State B (insufficient): 人为移除关键 evidence（候选 = 不相关 chunk）
                          → 期望 RETRIEVE（或 DECOMPOSE，取决于问题）
  State C (decompose)   : 候选只覆盖两个子问题中的一个
                          → 期望 DECOMPOSE（问题含两个独立子问题）

如果同一问题因 state 不同而采取不同 action → 证明 policy 是 state-dependent 的，
不是固定规则。

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step105_counterfactual.py

产出: eval_results/step105_counterfactual_<timestamp>.json
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

from src.agentic_rag import AgenticRAG  # noqa: E402

# 每个测试问题的 3 种 state 定义
CASES = [
    {
        "id": "cf_01",
        "question": "Transformer 编码器由几个相同的层堆叠而成，每层包含哪两个子层？",
        "states": {
            "sufficient": {
                "candidates": [
                    {
                        "id": "gold_a1",
                        "text": "Transformer 编码器由 N=6 个相同的层堆叠而成，每层包含两个子层：1. 多头自注意力子层 2. 逐位置的前馈神经网络（FFN）子层。每个子层后使用残差连接和层归一化（Layer Normalization）。",
                    },
                ],
                "expected": "ACCEPT",
            },
            "insufficient": {
                "candidates": [
                    {
                        "id": "irr_1",
                        "text": "肺栓塞（Pulmonary Embolism）的病理生理是一个动态过程，根据病程主要分为急性和慢性两种状态。",
                    },
                ],
                "expected": "RETRIEVE",
            },
        },
    },
    {
        "id": "cf_02",
        "question": "急性肺栓塞和慢性肺栓塞在病理生理上有什么区别？CTPA 影像上如何鉴别急性血栓与慢性血栓？",
        "states": {
            "sufficient": {
                "candidates": [
                    {
                        "id": "gold_b1",
                        "text": "急性肺栓塞：核心病理特征是血栓突然阻塞肺动脉或其分支，导致急性血流动力学障碍和气体交换异常。慢性肺栓塞：急性肺栓塞若未及时治疗，血栓机化转化为慢性血栓栓塞性肺动脉高压（CTEPH）。",
                    },
                    {
                        "id": "gold_b2",
                        "text": "急慢性血栓鉴别：新鲜血栓常见于急性期，如马鞍征、环征/轨道征；陈旧血栓表现为钝角偏心性附壁充盈缺损、纤维蹼样征。",
                    },
                ],
                "expected": "ACCEPT",
            },
            "partial": {
                # 只覆盖第一个子问题（病理生理），缺第二个（CTPA 影像鉴别）的关键信息
                "candidates": [
                    {
                        "id": "gold_b1",
                        "text": "急性肺栓塞：核心病理特征是血栓突然阻塞肺动脉或其分支，导致急性血流动力学障碍和气体交换异常。慢性肺栓塞：急性肺栓塞若未及时治疗，血栓机化转化为CTEPH。",
                    },
                ],
                "expected": "DECOMPOSE",
            },
        },
    },
]

# 无 LLM 时用规则 fallback（grade 基于 compute_relevance + 词面重叠）
# → 足够验证 decision 映射（sufficient→ACCEPT, insufficient→RETRIEVE）
# → 但 needs_decomposition 判定在规则 fallback 里不产生（规则只会出 sufficient/insufficient）
# → 所以 cf_02 partial 状态需要 LLM grader 才能测 DECOMPOSE。分两档跑。


def main():
    print("=" * 70)
    print("  🔬 Step 10.5: Counterfactual Policy Test")
    print("=" * 70, flush=True)

    # 用真实 retriever/generator（如果可用）构造 Agent
    from src.embeddings import get_embedding_provider
    from src.generator import create_generator
    from src.milvus_store import MilvusStore
    from src.reranker import CrossEncoderReranker
    from src.retriever import Retriever

    provider = get_embedding_provider("local")
    provider.warmup()
    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=5,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="memory",  # 不依赖磁盘索引（counterfactual 用手工候选）
    )
    generator = create_generator()
    agent = AgenticRAG(retriever=retriever, generator=generator, reranker=reranker, max_iterations=2)

    results = []
    for case in CASES:
        print(f"\n── {case['id']}: {case['question'][:50]}")
        for state_name, state_cfg in case["states"].items():
            # 直接构造 AgentState 注入候选，调 policy 判定（不跑完整检索）
            from src.agentic_rag import AgentState

            state = AgentState(original_query=case["question"])
            state.candidates = state_cfg["candidates"]
            state.iteration = 1  # 模拟已检索一轮

            # 用 evidence_grade 模拟"当前 state 的评分" + policy 决策
            grade = agent.evidence_grade(case["question"], state.candidates)
            status, decision, pmode = agent.policy(case["question"], state, grade)

            # 评估:决策是否匹配期望
            expected = state_cfg["expected"]
            match = decision == expected
            print(
                f"  [{state_name}] grade={grade['decision']}({grade['mode']}) → {decision} "
                f"(mode={pmode}) (期望 {expected}, {'✅' if match else '❌'})"
            )

            results.append(
                {
                    "case": case["id"],
                    "state": state_name,
                    "grade": grade["decision"],
                    "mode": grade["mode"],
                    "policy_mode": pmode,
                    "status": status,
                    "decision": decision,
                    "expected": expected,
                    "match": match,
                }
            )

    n_match = sum(1 for r in results if r["match"])
    n_total = len(results)
    print(f"\n  📊 Counterfactual 一致性: {n_match}/{n_total}")
    for r in results:
        print(f"    {'✅' if r['match'] else '❌'} {r['case']}/{r['state']}: {r['decision']} (期望 {r['expected']})")

    out = OUT_DIR / f"step105_counterfactual_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "n_match": n_match, "n_total": n_total}, f, ensure_ascii=False, indent=2)
    print(f"  📄 报告: {out}")


if __name__ == "__main__":
    main()

"""
Step 7: Rerank Query Oracle + End-to-End Rewrite Validation

7A  评测集清洗：对全部 40 个有 Gold 的 exact_match 题，检索每个 document 的所有 chunk，
    用 expected_answer_keywords 做 answer-bearing 判定：
      A = 该文档存在 answer-bearing chunk（chunk 内命中 ≥2 个关键词 或 ≥1 个强关键词）
      B = 部分承载（命中 1 个关键词）
      C = 文档正确但无 answer-bearing chunk
    （C 类不进入 Reranker 评测，避免"要求模型把不含答案的 chunk 排前面"）

7B/7C  Constraint-Preserving Rerank Query Augmentation：
      q_rerank = original + canonical_terms + must_preserve
      从 Original Query 强制提取：数字 / 比较符 / 单位 / 限定词 / 否定 / 缩写（must_preserve）
      canonical_terms 来自 V2 的 sparse_terms（LLM 已生成的中英术语）
      原则：绝不替换 Original Query

7D  冻结 Candidate Pool（不重新 retrieval），只测 Query：
      R0  = q_original
      R2  = q_original + canonical_terms
      R2C = q_original + canonical_terms + explicit must_preserve
      指标：Gold Rank / ΔRank / MRR / Hit@5 / Rescue@5 / Harm@5 / NetUtility@5
      用全部 answer-bearing A/B 题（不只 6 题）

7F  修复 Novelty Budget（Deeper retrieval）：
      V2 Dense Top100 + V2 BM25 Top100 → dedup → remove Original Top10 → Novelty Pool
      N5 / N10 / N20 / N40 / N80 与 q_rerank 组合跑 End-to-End Oracle
      （修复 Step 5 的 B40 bug：novelty 池被 dedup 限制在 ~26 个）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step7_rerank_query_oracle.py

产出: eval_results/step7_rerank_query_oracle_<timestamp>.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

from eval.test_questions import get_test_questions  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 5
FETCH_K = 20
RERANK_K = 10
DEEP = 100
NOVELTY_BUDGETS = [0, 5, 10, 20, 40, 80]

# ── 7C Constraint Extractor（规则版，不依赖 LLM）──
_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:%|mg|mmHg|mL|mm|cm|kg|岁|年|小时|天|周|月|h|s)?", re.IGNORECASE)
_UNIT_RE = re.compile(r"\b(mg|mmHg|mL|mL/min/1\.73m²|mm|cm|kg|Gy|HU|ms)\b", re.IGNORECASE)
_CONSTRAINT_WORDS = [
    "年龄校正",
    "age-adjusted",
    "重度",
    "severe",
    "长期",
    "chronic",
    "急性",
    "acute",
    "既往",
    "prior",
    "亚组",
    "subgroup",
    "校正",
    "adjusted",
    "校正后",
    "未",
    "无",
    "不",
    "非",
    "否定",
]
_CMP_RE = re.compile(r"[<>=≤≥]|小于|大于|不超过|至少|以上|以下|之间")
_ACRONYMS = [
    "FGSM",
    "sPESI",
    "YEARS",
    "HFpEF",
    "eGFR",
    "CTPA",
    "DICOM",
    "NIfTI",
    "LUNA16",
    "D-dimer",
    "U-Net",
    "RAG",
    "CT",
    "MRI",
    "DVT",
    "PE",
    "AUC",
    "HU",
    "Wells",
    "PESI",
    "FGSM",
    "CNN",
    "RNN",
    "LSTM",
    "GPU",
    "API",
    "SSD",
]


def extract_constraints(query: str) -> dict:
    """从 Original Query 提取 must_preserve 约束 + canonical 缩写"""
    nums = list(set(_NUM_RE.findall(query)))
    units = list(set(_UNIT_RE.findall(query)))
    cmps = list(set(_CMP_RE.findall(query)))
    constraint_words = [w for w in _CONSTRAINT_WORDS if w in query]
    acros = [a for a in _ACRONYMS if re.search(rf"\b{a}\b", query, re.IGNORECASE)]
    # 中文医学实体（看起来像术语的连续中文）
    entities = re.findall(r"[一-鿿]{2,8}", query)
    med_terms = [
        e
        for e in entities
        if any(
            k in e
            for k in [
                "校正",
                "重采样",
                "插值",
                "肺栓塞",
                "窗宽",
                "窗位",
                "结节",
                "分割",
                "训练",
                "损失",
                "流式",
                "检索",
                "评估",
            ]
        )
    ]
    return {
        "numbers": nums,
        "units": units,
        "comparators": cmps,
        "constraint_words": constraint_words,
        "acronyms": acros,
        "med_terms": med_terms,
    }


def build_q_rerank(q_original: str, constraints: dict, sparse_terms: str = "", add_must_preserve: bool = False) -> str:
    """Constraint-Preserving Rerank Query Augmentation

    绝不替换 Original Query：q_rerank = original + canonical + (must_preserve)
    """
    parts = [q_original]
    canonical = []
    if sparse_terms:
        canonical.append(sparse_terms)  # V2 已生成的中英术语（canonical_terms 来源）
    else:
        # 无 V2 时退化为缩写展开
        canonical.append(" ".join(constraints["acronyms"]))
    if canonical:
        parts.append(" | Canonical: " + " ".join(canonical))
    if add_must_preserve:
        mp = []
        if constraints["numbers"]:
            mp.append("numbers: " + ", ".join(constraints["numbers"]))
        if constraints["units"]:
            mp.append("units: " + ", ".join(constraints["units"]))
        if constraints["constraint_words"]:
            mp.append("constraints: " + ", ".join(constraints["constraint_words"]))
        if constraints["acronyms"]:
            mp.append("acronyms: " + ", ".join(constraints["acronyms"]))
        if mp:
            parts.append(" | MustPreserve: " + "; ".join(mp))
    return " ".join(parts)


def gold_filename(expected: str) -> str:
    return expected.rsplit(".", 1)[0]


def is_gold_chunk(r: dict, expected: str) -> bool:
    fn = r["metadata"].get("filename", "")
    return expected == fn or fn == gold_filename(expected)


def gold_rank(sources: list[dict], expected: str) -> int | None:
    if not expected:
        return None
    for i, s in enumerate(sources):
        if is_gold_chunk(s, expected):
            return i + 1
    return None


def dedup(results: list[dict]) -> list[dict]:
    seen: set = set()
    out = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def main():
    print("=" * 70)
    print("  🔬 Step 7: Rerank Query Oracle + End-to-End Rewrite Validation")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)}）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    reranker = CrossEncoderReranker()
    reranker._load_model()
    print(f"  ✅ Reranker ready={reranker.model_ready}", flush=True)

    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=TOP_K,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )

    questions = [q for q in get_test_questions() if q.get("expected_doc")]
    print(f"  📝 有 Gold 的题: {len(questions)}", flush=True)

    # ═══════════════════════════════════════════════
    #  7A: Answer-Bearing 清洗（按 document 判定）
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  🧹 7A: Answer-Bearing 清洗（按 document 判定）")
    print("=" * 70, flush=True)

    # 预建 filename → chunks 映射
    doc_chunks: dict[str, list[dict]] = {}
    for doc in all_docs:
        fn = doc["metadata"].get("filename", "")
        doc_chunks.setdefault(fn, []).append(doc)

    ab_quality = {}
    for q in questions:
        expected = q["expected_doc"]
        fn_base = gold_filename(expected)
        # 收集该 document 的所有 chunk
        chunks = doc_chunks.get(expected, []) + doc_chunks.get(fn_base, [])
        if not chunks:
            ab_quality[q["id"]] = "C"
            continue
        # 用 expected_answer_keywords 判 answer-bearing
        kws = q.get("expected_answer_keywords", [])
        if not kws:
            ab_quality[q["id"]] = "U"
            continue
        best_hits = 0
        for c in chunks:
            text = c["text"]
            hits = sum(1 for kw in kws if kw.lower() in text.lower())
            best_hits = max(best_hits, hits)
        if best_hits >= max(2, len(kws) // 2):
            ab_quality[q["id"]] = "A"
        elif best_hits >= 1:
            ab_quality[q["id"]] = "B"
        else:
            ab_quality[q["id"]] = "C"

    ab_stats = {}
    for qid, ql in ab_quality.items():
        ab_stats[ql] = ab_stats.get(ql, 0) + 1
    print(f"  Answer-bearing 分布: {ab_stats}", flush=True)
    print("  C 类（文档对但无 answer-bearing chunk，不进 Rerank 评测）:", flush=True)
    for q in questions:
        if ab_quality[q["id"]] == "C":
            print(f"    {q['id']:<12} {q['question'][:40]}", flush=True)

    # 有效 Gold 集（A + B）
    valid_questions = [q for q in questions if ab_quality[q["id"]] in ("A", "B")]
    print(f"  ✅ 有效 answer-bearing Gold: {len(valid_questions)} 题", flush=True)

    # ═══════════════════════════════════════════════
    #  7D: 冻结 Candidate Pool，只测 Query（R0 / R2 / R2C）
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  🔬 7D: 冻结 Candidate Pool 的 Query Oracle (R0 / R2 / R2C)")
    print("=" * 70, flush=True)

    # 从 Step 4 拿 V2 transformation（dense_query/sparse_terms）
    v2_files = sorted(OUT_DIR.glob("step4_v2_oracle_*.json"))
    step4_map = {}
    if v2_files:
        step4 = json.load(open(v2_files[-1], encoding="utf-8"))
        for c in step4["cases"]:
            step4_map[c["question"]] = c

    # 冻结候选池：Original Top10 + V2 Top40（B40 规模的 novelty）
    frozen_cases = []
    for q in valid_questions:
        question, expected = q["question"], q["expected_doc"]
        # Original Top10（冻结）
        orig_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:RERANK_K]
        orig_ids = {r["id"] for r in orig_cands}
        # V2 novelty（若该题有 transformation）
        dq = step4_map.get(question, {}).get("dense_query", "")
        st = step4_map.get(question, {}).get("sparse_terms", "")
        novelty = []
        if dq:
            vd = store.similarity_search(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=40)
            novelty = dedup(list(vd))
        if st:
            vb = bm25.search(st, top_k=40)
            novelty = dedup(list(novelty) + list(vb))
        novelty = [r for r in novelty if r["id"] not in orig_ids][:40]
        pool = dedup(list(orig_cands) + list(novelty))

        # 每个 query 变体
        constraints = extract_constraints(question)
        q_r2 = build_q_rerank(question, constraints, sparse_terms=st, add_must_preserve=False)
        q_r2c = build_q_rerank(question, constraints, sparse_terms=st, add_must_preserve=True)

        case = {
            "question": question,
            "expected": expected,
            "ab_quality": ab_quality[q["id"]],
            "pool_size": len(pool),
            "r0_gold_rank": gold_rank(pool, expected),
            "constraints": constraints,
        }
        # R0 / R2 / R2C：每个 rerank 一次
        for name, rq in [("r0", question), ("r2", q_r2), ("r2c", q_r2c)]:
            scored = reranker.rerank(rq, list(pool), len(pool))
            case[f"{name}_gold_rank"] = gold_rank(scored, expected)
        frozen_cases.append(case)
        if len(frozen_cases) % 8 == 0:
            print(f"    ...{len(frozen_cases)}/{len(valid_questions)}", flush=True)

    # ── 汇总（全有效 Gold）──
    def agg_query_results(cases, key):
        hits = sum(1 for c in cases if c[f"{key}_gold_rank"] is not None and c[f"{key}_gold_rank"] <= 5)
        mrr = sum(1 / c[f"{key}_gold_rank"] for c in cases if c[f"{key}_gold_rank"]) / max(len(cases), 1)
        mean_rank = sum(c[f"{key}_gold_rank"] or 100 for c in cases) / max(len(cases), 1)
        return {"hit5": hits, "mrr": round(mrr, 4), "mean_rank": round(mean_rank, 2)}

    r0_agg = agg_query_results(frozen_cases, "r0")
    r2_agg = agg_query_results(frozen_cases, "r2")
    r2c_agg = agg_query_results(frozen_cases, "r2c")
    # Rescue / Harm
    r2_rescue = sum(
        1
        for c in frozen_cases
        if c["r0_gold_rank"] and c["r0_gold_rank"] > 5 and c["r2_gold_rank"] and c["r2_gold_rank"] <= 5
    )
    r2_harm = sum(
        1
        for c in frozen_cases
        if c["r0_gold_rank"] and c["r0_gold_rank"] <= 5 and (c["r2_gold_rank"] is None or c["r2_gold_rank"] > 5)
    )
    r2c_rescue = sum(
        1
        for c in frozen_cases
        if c["r0_gold_rank"] and c["r0_gold_rank"] > 5 and c["r2c_gold_rank"] and c["r2c_gold_rank"] <= 5
    )
    r2c_harm = sum(
        1
        for c in frozen_cases
        if c["r0_gold_rank"] and c["r0_gold_rank"] <= 5 and (c["r2c_gold_rank"] is None or c["r2c_gold_rank"] > 5)
    )

    print("\n  📊 7D Query Oracle（全部有效 answer-bearing Gold）")
    print(f"  {'Variant':<8}{'Hit@5':>7}{'MRR':>8}{'MeanRank':>10}{'Rescue@5':>9}{'Harm@5':>8}{'NetUtil':>8}")
    print(f"  {'R0':<8}{r0_agg['hit5']:>7}{r0_agg['mrr']:>8}{r0_agg['mean_rank']:>10}{'-':>9}{'-':>8}{'-':>8}")
    print(
        f"  {'R2':<8}{r2_agg['hit5']:>7}{r2_agg['mrr']:>8}{r2_agg['mean_rank']:>10}{r2_rescue:>9}{r2_harm:>8}{r2_rescue - r2_harm:>8}"
    )
    print(
        f"  {'R2C':<8}{r2c_agg['hit5']:>7}{r2c_agg['mrr']:>8}{r2c_agg['mean_rank']:>10}{r2c_rescue:>9}{r2c_harm:>8}{r2c_rescue - r2c_harm:>8}"
    )

    # ═══════════════════════════════════════════════
    #  7F: 修复的 Novelty Budget（Deeper retrieval + q_rerank）
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  🔬 7F: End-to-End Novelty Budget（V2 Top100 + q_rerank）")
    print("=" * 70, flush=True)

    budget_cases = []
    for q in valid_questions:
        question, expected = q["question"], q["expected_doc"]
        dq = step4_map.get(question, {}).get("dense_query", "")
        st = step4_map.get(question, {}).get("sparse_terms", "")

        orig_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:RERANK_K]
        orig_ids = {r["id"] for r in orig_cands}

        # 修复版 Novelty Pool：V2 Dense Top100 + BM25 Top100 → dedup → 去 Original
        novelty = []
        if dq:
            vd = store.similarity_search(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=DEEP)
            novelty = dedup(list(vd))
        if st:
            vb = bm25.search(st, top_k=DEEP)
            novelty = dedup(list(novelty) + list(vb))
        novelty = [r for r in novelty if r["id"] not in orig_ids][:80]

        case = {
            "question": question,
            "expected": expected,
            "v0_cand_rank": gold_rank(orig_cands, expected),
            "novelty_pool_rank": gold_rank(novelty, expected),
            "novelty_pool_size": len(novelty),
            "budgets": {},
        }

        # q_rerank（constraint-preserving）
        constraints = extract_constraints(question)
        q_rerank = build_q_rerank(question, constraints, sparse_terms=st, add_must_preserve=True)

        # 一次性 rerank 完整 union（≤90），推导各 budget
        full_union = dedup(list(orig_cands) + list(novelty))
        scored = reranker.rerank(q_rerank, list(full_union), len(full_union))
        novelty_idx = {r["id"]: i for i, r in enumerate(novelty)}
        gold_ids_in_pool = [r["id"] for r in full_union if is_gold_chunk(r, expected)]

        for B in NOVELTY_BUDGETS:
            pool = dedup(list(orig_cands) + novelty[:B])
            in_budget = [r for r in scored if r["id"] in orig_ids or novelty_idx.get(r["id"], 1 << 30) < B][:TOP_K]
            rank = gold_rank(in_budget, expected)
            cand_rescue = case["v0_cand_rank"] is None and gold_rank(pool, expected) is not None
            case["budgets"][f"N{B}"] = {
                "pool_size": len(pool),
                "cand_rescue": cand_rescue,
                "rerank_rank": rank,
                "rerank_rescue": rank is not None,
            }
        budget_cases.append(case)
        if len(budget_cases) % 8 == 0:
            print(f"    ...{len(budget_cases)}/{len(valid_questions)}", flush=True)

    # ── 汇总 ──
    agg = {}
    for B in NOVELTY_BUDGETS:
        cand = sum(1 for c in budget_cases if c["budgets"][f"N{B}"]["cand_rescue"])
        final_r = sum(1 for c in budget_cases if c["budgets"][f"N{B}"]["rerank_rescue"])
        harm = 0
        for c in budget_cases:
            # guardrail = R0 有效 Gold ≤5（用 r0 冻结候选 rank）
            if c["v0_cand_rank"] is not None and c["v0_cand_rank"] <= 5:
                if not c["budgets"][f"N{B}"]["rerank_rescue"]:
                    harm += 1
        avg_pool = sum(c["budgets"][f"N{B}"]["pool_size"] for c in budget_cases) / max(len(budget_cases), 1)
        agg[f"N{B}"] = {
            "candidate_rescue": cand,
            "final_rescue": final_r,
            "harm": harm,
            "net_utility": final_r - harm,
            "avg_pool": round(avg_pool, 1),
        }

    print("\n  📊 7F Novelty Budget（V2 Top100 + q_rerank constraint-preserving）")
    print(f"  {'Budget':<7}{'CandRescue':>11}{'FinalRescue':>12}{'Harm':>6}{'NetUtil':>8}{'AvgPool':>9}")
    for B in NOVELTY_BUDGETS:
        a = agg[f"N{B}"]
        print(
            f"  {'N' + str(B):<7}{a['candidate_rescue']:>11}{a['final_rescue']:>12}{a['harm']:>6}{a['net_utility']:>8}{a['avg_pool']:>9}"
        )

    # 最终总表（对应 Step 7G 的完整故事）
    print("\n" + "=" * 70)
    print("  📊 最终总表：整个 Rewrite 实验")
    print("=" * 70)
    print("  Old Paraphrase + multi-RRF : CandidateRescue=0 Final=0 Harm=4 (Step 2)")
    print("  V2 + Original rerank query : CandidateRescue=4 Final=0 Harm=1 (Step 5)")
    print(
        f"  V2 + q_rerank (7D frozen)   : CandidateRescue=? Final={r2c_agg['hit5']} Rescue={r2c_rescue} Harm={r2c_harm}"
    )
    print(
        f"  V2 + q_rerank + deep (7F)   : CandidateRescue={agg['N40']['candidate_rescue']} Final={agg['N40']['final_rescue']} Harm={agg['N40']['harm']} NetUtility={agg['N40']['net_utility']}"
    )

    out = OUT_DIR / f"step7_rerank_query_oracle_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ab_quality": ab_quality,
                "ab_stats": ab_stats,
                "valid_questions": [q["id"] for q in valid_questions],
                "frozen_cases": frozen_cases,
                "r0_agg": r0_agg,
                "r2_agg": r2_agg,
                "r2c_agg": r2c_agg,
                "r2_rescue": r2_rescue,
                "r2_harm": r2_harm,
                "r2c_rescue": r2c_rescue,
                "r2c_harm": r2c_harm,
                "budget_agg": agg,
                "budget_cases": budget_cases,
                "elapsed": round(time.time() - _t0, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out}")


_t0 = time.time()
if __name__ == "__main__":
    main()

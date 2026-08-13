"""
Step 4: Retriever-aware Transformation Oracle（V2 先跑）

在 SUPPORTED_RETRIEVAL_MISS 子集（Step 3 中 Gold 在语料 rank<=100 但 Original Top10 miss 的题，
共 13 题：B 类 6 + D 类 7）上验证：
  Retriever-aware 双查询（dense_query 给 Dense / sparse_terms 给 BM25）能否 Rescue Gold。

Oracle 协议（Step 4D — Protected Candidate Union）：
  Original Top10 ──────────┐
                           ├── UNION + DEDUP（≤30）── rerank(q_original) ── Top5
  Transformed Top10（Dense+BM25）┘
  ✗ 不做跨 query 二次 RRF —— 隔离 "Transformation 本身没用" 与 "Fusion 把它毁了" 两种失败

对照 V0 Baseline：Original Only → Dense+BM25 → RRF → rerank → Top5（无改写）

核心指标：Rescue@5 / Harm@5 / NetUtility@5 = Rescue − Harm
（V1 Current Paraphrase 的已知数据：Rescue@5 = 0, Harm@5 = 4，作为负面对照）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step4_v2_oracle.py

产出: eval_results/step4_v2_oracle_<timestamp>.json
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

from src.embeddings import get_embedding_provider  # noqa: E402
from src.generator import create_generator  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 5  # 生产输出
FETCH_K = 20  # 生产 fetch_k = max(top_k*2, 20)
RERANK_K = 10  # 候选截断
UNION_LIMIT = 30

# ── V2 Transformation Prompt ──────────────────────────

V2_SYSTEM_PROMPT = """\
你是一个医学检索查询转换助手，服务于医学影像 / 肺栓塞 / 深度学习医学应用知识库的混合检索系统。

你的任务：把用户的原始问题转换成两条"检索器感知"的查询：

1. dense_query — 用于语义向量检索（Dense）。要求：
   - 语义完整、自然语言（中英文均可，corpus 是中英文学术资料）
   - 补全缩写全称与上下文，如 sPESI → simplified Pulmonary Embolism Severity Index

2. sparse_terms — 用于关键词检索（BM25）。要求：
   - 医学关键词列表，逗号分隔：缩写 + 全称、中英术语、数字、实体名
   - 绝对不允许删除原始问题中的实体词（缩写、疾病名、检查名、数字）

判断：如果问题明显超出知识库范围（非医学影像/肺栓塞/深度学习医学应用），
输出 {"unsupported": true} 并说明原因，此时 dense_query / sparse_terms 为空字符串。

只输出一个 JSON 对象，不要其他内容。

## 示例

原始问题：sPESI评分包含哪些评估项目？
输出：
{"dense_query": "What clinical variables and criteria are included in the simplified Pulmonary Embolism Severity Index (sPESI) score?", "sparse_terms": "sPESI, simplified Pulmonary Embolism Severity Index, 肺栓塞严重程度指数, score, criteria, variables, 评分项目", "unsupported": false}

原始问题：体素重采样中三线性插值和最近邻插值各有什么优缺点？
输出：
{"dense_query": "Comparison of trilinear interpolation and nearest neighbor interpolation in voxel resampling for medical image preprocessing", "sparse_terms": "体素重采样, voxel resampling, trilinear interpolation, 三线性插值, nearest neighbor, 最近邻插值, NIfTI, DICOM", "unsupported": false}

原始问题：2025年全球经济增长率是多少？
输出：
{"dense_query": "", "sparse_terms": "", "unsupported": true, "reason": "非医学影像/肺栓塞/深度学习医学应用领域"}"""

V2_USER_PROMPT = "原始问题：{query}\n输出 JSON："


def transform_v2(llm, query: str) -> dict | None:
    """调用 LLM 生成 (dense_query, sparse_terms)；失败/解析失败返回 None"""
    try:
        response = llm.chat(
            messages=[
                {"role": "system", "content": V2_SYSTEM_PROMPT},
                {"role": "user", "content": V2_USER_PROMPT.format(query=query)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
    except Exception:
        return None
    m = re.search(r"\{[\s\S]*\}", str(response))
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if data.get("unsupported"):
        return {"unsupported": True, "reason": data.get("reason", "")}
    dq = str(data.get("dense_query", "")).strip()
    st = str(data.get("sparse_terms", "")).strip()
    if not dq or not st:
        return None
    return {"dense_query": dq, "sparse_terms": st, "unsupported": False}


def gold_rank(sources: list[dict], expected: str) -> int | None:
    """Gold 在结果中的 rank（1-based），不在则 None"""
    if not expected:
        return None
    expected_base = expected.rsplit(".", 1)[0]
    for i, s in enumerate(sources):
        fn = s["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            return i + 1
    return None


def dedup(results: list[dict]) -> list[dict]:
    """按 id 去重，保持顺序"""
    seen: set = set()
    out = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def main():
    print("=" * 70)
    print("  🔬 Step 4: Retriever-aware Transformation Oracle (V2)")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    if bm25.get_total_docs() != len(all_docs):
        print(f"  🆕 重建 BM25 索引（{len(all_docs)}）...", flush=True)
        bm25.rebuild(all_docs)
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    reranker = CrossEncoderReranker()

    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=TOP_K,
        generator=generator,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )

    # ── SUPPORTED_RETRIEVAL_MISS 子集：从 Step 3 报告提取 ──
    diag_files = sorted(OUT_DIR.glob("rewrite_failure_diagnosis_*.json"))
    if not diag_files:
        print("  ❌ 未找到 rewrite_failure_diagnosis 报告，先跑 Step 3")
        return
    diag = json.load(open(diag_files[-1], encoding="utf-8"))

    # B + D 类（有 Gold 标注且 rank<=100 找到）去重；C ⊂ B 不重复计数
    supported: dict[str, dict] = {}
    for case in diag.get("B_retrieval_miss", []) + diag.get("D_other", []):
        q = case["question"]
        if case.get("expected") and q not in supported:
            supported[q] = case
    questions = [{"question": q, "expected": c["expected"], "diag": c} for q, c in supported.items()]
    print(
        f"  📝 SUPPORTED_RETRIEVAL_MISS: {len(questions)} 题（Gold 在语料 Top100，Original Top10 miss）\n", flush=True
    )

    # 40 个 A 类的 gold 标注核查
    n_A_without_gold = sum(1 for c in diag.get("A_corpus_absent", []) if not c.get("expected"))
    print(f"  ⚠️  A 类 40 题中 {n_A_without_gold} 题无 expected_doc —— corpus-level 存在性无法验证\n", flush=True)

    # ── 逐题 Oracle ──
    cases = []
    stats = {
        "N": 0,
        "V0_top5_hit": 0,
        "V2_top5_hit": 0,
        "Rescued@5": 0,
        "Harmed@5": 0,
        "new_candidate_recall@10": 0,
        "unsupported_judged": 0,
    }
    rank_changes = []  # V2 最终 rank − V0 候选 rank（both-found）

    t0 = time.time()
    for idx, q in enumerate(questions, 1):
        question, expected = q["question"], q["expected"]
        print(f"  ── [{idx}/{len(questions)}] {question[:42]}", flush=True)
        stats["N"] += 1

        case = {
            "question": question,
            "expected": expected,
            "diag_ranks": {
                k: q["diag"].get(k)
                for k in (
                    "dense_rank",
                    "bm25_rank",
                    "rrf_rank",
                    "rewrite1_rrf_rank",
                    "rewrite2_rrf_rank",
                    "rewrite3_rrf_rank",
                )
            },
        }

        # ── V0 Baseline：Original Only（fetch_k=20 → 候选10 → rerank → Top5）──
        orig_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:RERANK_K]
        case["v0_cand_rank"] = gold_rank(orig_cands, expected)
        v0_top5 = reranker.rerank(question, list(orig_cands), TOP_K)
        case["v0_top5_rank"] = gold_rank(v0_top5, expected)
        v0_hit = case["v0_top5_rank"] is not None
        stats["V0_top5_hit"] += v0_hit

        # ── V2 Transformation ──
        t_call = time.time()
        tf = transform_v2(generator, question)
        case["transform_time_s"] = round(time.time() - t_call, 1)
        if tf is None:
            case["transform_failed"] = True
            case["v2_top5_hit"] = False
            cases.append(case)
            continue
        if tf.get("unsupported"):
            stats["unsupported_judged"] += 1
            case["unsupported"] = True
            case["reason"] = tf.get("reason", "")

        dense_query, sparse_terms = tf["dense_query"], tf["sparse_terms"]
        case["dense_query"] = dense_query
        case["sparse_terms"] = sparse_terms
        print(f"      Dense: {dense_query[:70]}", flush=True)
        print(f"      BM25 : {sparse_terms[:70]}", flush=True)

        # Dense-transformed rank@10
        q_emb = provider.embed([dense_query], prefix="query: ")[0]
        dense_top10 = store.similarity_search(query_embedding=q_emb, top_k=10)
        case["v2_dense_rank@10"] = gold_rank(dense_top10, expected)

        # BM25-transformed rank@10（sparse_terms 逗号分隔 → 空格分词交给 Whoosh）
        bm25_top10 = bm25.search(sparse_terms, top_k=10)
        case["v2_bm25_rank@10"] = gold_rank(bm25_top10, expected)

        # ── Protected Union：Original Top10 ∪ Dense Top10 ∪ BM25 Top10 ──
        union = dedup(list(orig_cands) + list(dense_top10) + list(bm25_top10))[:UNION_LIMIT]
        case["union_size"] = len(union)
        case["union_gold_rank"] = gold_rank(union, expected)

        # rerank(q_original) → Top5
        v2_top5 = reranker.rerank(question, list(union), TOP_K)
        case["v2_top5_rank"] = gold_rank(v2_top5, expected)
        v2_hit = case["v2_top5_rank"] is not None
        stats["V2_top5_hit"] += v2_hit

        if v2_hit and not v0_hit:
            stats["Rescued@5"] += 1
            print(f"      🆘 RESCUED! V2 rank={case['v2_top5_rank']}", flush=True)
        elif v0_hit and not v2_hit:
            stats["Harmed@5"] += 1
            print(f"      💥 HARMED (V0 rank={case['v0_top5_rank']})", flush=True)
        elif v2_hit and v0_hit:
            stats["Rescued@5"] += 0

        # New Candidate Recall@10：V0 候选无 Gold 但 Union 有
        if case["v0_cand_rank"] is None and case["union_gold_rank"] is not None:
            stats["new_candidate_recall@10"] += 1

        # ΔGoldRank（两路都找到时）
        if case["v0_cand_rank"] is not None and case["v2_top5_rank"] is not None:
            rank_changes.append(case["v2_top5_rank"] - case["v0_cand_rank"])

        cases.append(case)

    elapsed = time.time() - t0

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("  📊 V2 Oracle 结果")
    print("=" * 70)
    s = stats
    print(f"  N = {s['N']}")
    print(f"  V0 Baseline Top5 hit     = {s['V0_top5_hit']}")
    print(f"  V2 Top5 hit              = {s['V2_top5_hit']}")
    print(f"  🆘 Rescued@5             = {s['Rescued@5']}")
    print(f"  💥 Harmed@5              = {s['Harmed@5']}")
    print(f"  📈 NetUtility@5          = {s['Rescued@5'] - s['Harmed@5']:+d}")
    print(f"  🆕 New Candidate Recall@10 = {s['new_candidate_recall@10']}")
    if rank_changes:
        print(f"  ΔGoldRank (V2−V0, both-found): mean = {sum(rank_changes) / len(rank_changes):+.2f}")
    print(f"  LLM 判定 unsupported 次数 = {s['unsupported_judged']}")
    print(f"  ⏱️  耗时 {elapsed:.0f}s")

    print("\n  📋 逐题明细:")
    for c in cases:
        d = c["diag_ranks"]
        v0 = str(c.get("v0_cand_rank"))
        v0t = str(c.get("v0_top5_rank"))
        v2d = str(c.get("v2_dense_rank@10"))
        v2b = str(c.get("v2_bm25_rank@10"))
        v2t = str(c.get("v2_top5_rank"))
        tag = (
            "  RESCUED"
            if c.get("v2_top5_rank") is not None and c.get("v0_top5_rank") is None
            else "  HARMED"
            if c.get("v0_top5_rank") is not None and c.get("v2_top5_rank") is None
            else ""
        )
        print(
            f"    V0(cand={v0:>3},top5={v0t:>3}) V2(dense={v2d:>3},bm25={v2b:>3},top5={v2t:>3})"
            f" | diag: d={str(d['dense_rank']):>3} b={str(d['bm25_rank']):>3} rrf={str(d['rrf_rank']):>3}"
            f"{tag} | {c['question'][:34]}"
        )

    out = OUT_DIR / f"step4_v2_oracle_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "cases": cases,
                "delta_gold_rank_mean": round(sum(rank_changes) / len(rank_changes), 2) if rank_changes else None,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

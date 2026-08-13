"""
Step 6: Cross-Encoder Failure Anatomy

对 Step 5 的 4 个 RerankFailure case（B20 候选池有 Gold 但 rerank 后 >5）+ 2 个 guardrail case
做完整诊断：

6.0  B40 budget 生效性确认（pool_size 是否 > B20）
6.1  Gold + hard negatives 完整 raw-score trace（Gap@5 / Gap@1）
6.2  Answer-bearing check（Gold chunk 是否独立包含答案 —— 人工标记位）
6.3  Truncation check（query/passage token len vs max_length）
6.4  Hard Negative Taxonomy（H1-H5 分类）
6.5  三个 Rerank Oracle（冻结 pool）：
       R0 = q_original × chunk_text（当前 baseline）
       R1 = q_original × title+heading+chunk_text（metadata-aware）
       R2 = q_original+canonical × chunk_text（canonical rerank query）
       R3 = q_original × answer-window passage（仅诊断，找答案窗口）

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step6_reranker_anatomy.py

产出: eval_results/step6_reranker_anatomy_<timestamp>.json
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
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.reranker import CrossEncoderReranker  # noqa: E402
from src.retriever import Retriever  # noqa: E402

TOP_K = 5
FETCH_K = 20
RERANK_K = 10
BUDGETS = [0, 5, 10, 20, 40]


def gold_rank(sources: list[dict], expected: str) -> int | None:
    if not expected:
        return None
    expected_base = expected.rsplit(".", 1)[0]
    for i, s in enumerate(sources):
        fn = s["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            return i + 1
    return None


def gold_ids(chunks: list[dict], expected: str) -> list[str]:
    """返回所有匹配 expected 的 chunk id（可能一个 doc 多个 chunk 命中）"""
    if not expected:
        return []
    expected_base = expected.rsplit(".", 1)[0]
    out = []
    for r in chunks:
        fn = r["metadata"].get("filename", "")
        if expected == fn or fn == expected_base:
            out.append(r["id"])
    return out


def dedup(results: list[dict]) -> list[dict]:
    seen: set = set()
    out = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def count_tokens(text: str) -> int:
    """粗略 token 估计（英文按词 + 中文按字 + 标点）"""
    # 用 tokenizer 会慢，这里用启发式：英文按空格分词 + 中文按字 + 数字
    words = re.findall(r"[A-Za-z0-9]+", text)
    chinese = re.findall(r"[一-鿿]", text)
    other = re.findall(r"[^\w一-鿿\s]", text)
    return len(words) + len(chinese) + len(other) // 2


def hard_neg_type(gold_text: str, hn_text: str) -> str:
    """Hard negative 分类 H1-H5"""
    # 同 document?
    # (通过外层比较 filename)
    # H3 Constraint violation: 数字/否定/条件词差异
    gold_nums = set(re.findall(r"\d+\.?\d*", gold_text))
    hn_nums = set(re.findall(r"\d+\.?\d*", hn_text))
    has_constraint_words = any(
        w in gold_text for w in ["年龄", "校正", "age-adjusted", "epsilon", "亚组", "eGFR", "<", ">", "不", "非"]
    )
    if has_constraint_words and (gold_nums - hn_nums):
        return "H3 Constraint Violation"
    # 简单启发式: 默认
    return "H1 Topic Match"


def main():
    print("=" * 70)
    print("  🔬 Step 6: Cross-Encoder Failure Anatomy")
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

    # ── 从 Step 4 报告取已冻结的 V2 transformation + guardrail ──
    step4_files = sorted(OUT_DIR.glob("step4_v2_oracle_*.json"))
    if not step4_files:
        print("  ❌ 未找到 step4_v2_oracle 报告")
        return
    step4 = json.load(open(step4_files[-1], encoding="utf-8"))
    # 每题: question → {expected, dense_query, sparse_terms, v0_top5_rank}
    q_map = {}
    for c in step4["cases"]:
        q_map[c["question"]] = {
            "expected": c["expected"],
            "dense_query": c.get("dense_query", ""),
            "sparse_terms": c.get("sparse_terms", ""),
            "v0_top5_rank": c.get("v0_top5_rank"),
        }

    # ── 从 Step 5 报告取 4 个 RerankFailure ──
    step5_files = sorted(OUT_DIR.glob("step5_candidate_budget_*.json"))
    if not step5_files:
        print("  ❌ 未找到 step5 报告")
        return
    step5 = json.load(open(step5_files[-1], encoding="utf-8"))

    rerank_failures = []  # B20 候选有 Gold 但 rerank 后 >5
    for c in step5["sweep_cases"]:
        b20 = c["budgets"].get("B20", {})
        if b20.get("cand_rescue") and not b20.get("rerank_rescue"):
            rerank_failures.append(c["question"])

    # guardrail: Step 4 中 V0 Top5 hit 的 2 题
    guardrail_cases = [q for q, info in q_map.items() if info["v0_top5_rank"] is not None]
    print(f"  🔍 RerankFailure cases ({len(rerank_failures)}):", flush=True)
    for q in rerank_failures:
        print(f"      {q}", flush=True)
    print(f"  🔍 Guardrail cases ({len(guardrail_cases)}):", flush=True)
    for q in guardrail_cases:
        print(f"      {q}", flush=True)

    all_cases = []
    case_questions = rerank_failures + guardrail_cases

    # 6.0 确认 B40 生效：打印 pool sizes
    print("\n  📊 6.0 B20 vs B40 pool size（Step 5 数据）", flush=True)
    for c in step5["sweep_cases"]:
        b20 = c["budgets"].get("B20", {})
        b40 = c["budgets"].get("B40", {})
        if c["question"] in case_questions:
            print(f"    {c['question'][:32]:<34} B20={b20.get('pool_size')} B40={b40.get('pool_size')}", flush=True)

    # ── 逐 case 分析 ──
    for q in case_questions:
        c = q_map[q]
        question, expected = q, c["expected"]
        dq, st = c.get("dense_query", ""), c.get("sparse_terms", "")
        print(f"\n{'=' * 70}\n  📋 CASE: {question[:50]}\n{'=' * 70}", flush=True)

        case = {"question": question, "expected": expected, "dense_query": dq, "sparse_terms": st}

        # 冻结 B20 候选池（Original Top10 + novelty[:20]）
        orig_cands = retriever._hybrid_retrieve(question, fetch_k=FETCH_K)[:RERANK_K]
        orig_ids = {r["id"] for r in orig_cands}
        v_dense = (
            store.similarity_search(query_embedding=provider.embed([dq], prefix="query: ")[0], top_k=40) if dq else []
        )
        v_bm25 = bm25.search(st, top_k=40) if st else []
        novelty = dedup(list(v_dense) + list(v_bm25))
        novelty = [r for r in novelty if r["id"] not in orig_ids][:40]
        pool = dedup(list(orig_cands) + novelty[:20])

        gold_expected = [g for g in gold_ids(pool, expected)]
        case["gold_chunk_ids"] = gold_expected
        case["pool_size"] = len(pool)

        # ── R0: 当前 baseline ──
        r0 = reranker.rerank(question, list(pool), len(pool))
        case["r0"] = {
            "scores": [
                {"id": r["id"], "score": r.get("_rerank_score", 0), "filename": r["metadata"].get("filename", "")}
                for r in r0
            ]
        }
        r0_gold_ranks = [i + 1 for i, r in enumerate(r0) if r["id"] in gold_expected]
        case["r0_gold_rank"] = r0_gold_ranks[0] if r0_gold_ranks else None
        print(f"  R0 Gold rank = {case['r0_gold_rank']}", flush=True)

        # Gap@5 / Gap@1
        gold_scores = [r.get("_rerank_score", 0) for r in r0 if r["id"] in gold_expected]
        if gold_scores:
            gs = max(gold_scores)
            r5s = r0[4].get("_rerank_score", 0) if len(r0) > 4 else None
            r1s = r0[0].get("_rerank_score", 0)
            case["gap@5"] = round(gs - r5s, 4) if r5s is not None else None
            case["gap@1"] = round(gs - r1s, 4)
            print(f"  R0 Gold score={gs:.4f}  Rank5={r5s}  Gap@5={case['gap@5']}  Gap@1={case['gap@1']}", flush=True)

        # Top hard negatives（Gold 之前的）
        top_hn = []
        for r in r0:
            if r["id"] in gold_expected:
                break
            top_hn.append(r)
        case["top_hard_negatives"] = [
            {
                "id": r["id"],
                "score": r.get("_rerank_score", 0),
                "filename": r["metadata"].get("filename", ""),
                "heading": r["metadata"].get("heading", ""),
                "text": r["text"][:200],
                "source": r.get("_retriever", ""),
            }
            for r in top_hn[:5]
        ]
        print(f"  Top hard negatives: {len(top_hn)} 个在 Gold 前", flush=True)
        for hn in case["top_hard_negatives"]:
            print(
                f"    rank{top_hn.index([x for x in top_hn if x['id'] == hn['id']][0]) + 1}: score={hn['score']:.4f} {hn['filename'][:30]} | {hn['text'][:60]}",
                flush=True,
            )

        # ── 6.2 Answer-bearing check（打印 Gold chunk 文本供人工判断）──
        gold_texts = []
        for r in pool:
            if r["id"] in gold_expected:
                gold_texts.append(r["text"])
        case["gold_chunk_texts"] = gold_texts
        print("\n  📋 6.2 Gold chunk 文本（判断是否 answer-bearing）:", flush=True)
        for i, gt in enumerate(gold_texts):
            print(f"    [{i}] {gt[:300]}...", flush=True)

        # ── 6.3 Truncation check ──
        case["max_length"] = (
            getattr(CrossEncoderReranker._model, "max_length", None) if CrossEncoderReranker._model else None
        )
        # bge-reranker-v2-m3 tokenizer
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", local_files_only=True)
            q_tokens = tok(question)["input_ids"]
            gold_tok_lens = [len(tok(t)["input_ids"]) for t in gold_texts]
            case["query_token_len"] = len(q_tokens)
            case["gold_token_lens"] = gold_tok_lens
            case["was_truncated"] = [l > 512 for l in gold_tok_lens]
            print(
                f"  📏 6.3 query_tokens={len(q_tokens)}  gold_tokens={gold_tok_lens}  max_length={case['max_length']}  truncated={case['was_truncated']}",
                flush=True,
            )
        except Exception as e:
            print(f"  ⚠️ tokenizer 不可用: {e}", flush=True)
            case["query_token_len"] = count_tokens(question)
            case["gold_token_lens"] = [count_tokens(t) for t in gold_texts]

        # ── 6.4 Hard negative taxonomy ──
        hn_taxonomy = []
        for hn in case["top_hard_negatives"]:
            gt = gold_texts[0] if gold_texts else ""
            t = hard_neg_type(gt, hn["text"])
            hn_taxonomy.append({"id": hn["id"], "type": t})
        case["hard_neg_taxonomy"] = hn_taxonomy
        print(f"  🏷️ 6.4 Hard negative types: {[t['type'] for t in hn_taxonomy]}", flush=True)

        # ── 6.5 Rerank Oracles（冻结 pool）──
        # R1: metadata-aware passage
        def meta_passage(r):
            fn = r["metadata"].get("filename", "")
            hd = r["metadata"].get("heading", "")
            sec = r["metadata"].get("section_title", "")
            return f"{fn}\n{hd}\n{sec}\n{r['text']}"

        r1_pairs = [(question, meta_passage(r)) for r in pool]
        r1_scores = reranker.rerank_pairs(r1_pairs)
        r1_rank = None
        for i, r in enumerate(pool):
            if r["id"] in gold_expected:
                r1_rank = i + 1
                break
        # 按 r1_scores 重新排序求 rank
        order1 = sorted(range(len(pool)), key=lambda i: r1_scores[i], reverse=True)
        r1_gold_rank = next((i + 1 for i, oi in enumerate(order1) if pool[oi]["id"] in gold_expected), None)
        case["r1_gold_rank"] = r1_gold_rank
        print(f"  R1 (title+heading+text) Gold rank = {r1_gold_rank}", flush=True)

        # R2: canonical rerank query = original + sparse_terms 的关键实体
        canonical = question + " " + (st or "")
        r2_pairs = [(canonical, r["text"]) for r in pool]
        r2_scores = reranker.rerank_pairs(r2_pairs)
        order2 = sorted(range(len(pool)), key=lambda i: r2_scores[i], reverse=True)
        r2_gold_rank = next((i + 1 for i, oi in enumerate(order2) if pool[oi]["id"] in gold_expected), None)
        case["r2_gold_rank"] = r2_gold_rank
        print(f"  R2 (canonical query) Gold rank = {r2_gold_rank}", flush=True)

        # R3: answer-window oracle（只找 Gold chunk 内得分最高的句子窗口）
        if gold_texts:
            gt = gold_texts[0]
            sentences = re.split(r"(?<=[。！？.!?])\s*", gt)
            windows = []
            for si in range(len(sentences)):
                window = " ".join(sentences[max(0, si - 1) : si + 2])
                windows.append(window)
            r3_pairs = [(question, w) for w in windows]
            r3_scores = reranker.rerank_pairs(r3_pairs)
            best_wi = max(range(len(windows)), key=lambda i: r3_scores[i])
            case["r3_best_window_score"] = r3_scores[best_wi]
            case["r3_best_window_text"] = windows[best_wi][:150]
            # 对比：完整 chunk 的分数
            full_score = r3_scores[-1] if False else None
            r0_gold_score = max((r.get("_rerank_score", 0) for r in r0 if r["id"] in gold_expected), default=None)
            case["r3_vs_full_gold_score"] = {"full_gold_score": r0_gold_score, "best_window_score": r3_scores[best_wi]}
            print(
                f"  R3 answer-window: best window score={r3_scores[best_wi]:.4f} vs full gold score={r0_gold_score}",
                flush=True,
            )
            print(f"      window: {windows[best_wi][:80]}...", flush=True)

        all_cases.append(case)

    # ── 汇总输出 ──
    print("\n" + "=" * 70)
    print("  📊 Step 6 汇总")
    print("=" * 70)
    print(f"  {'Case':<34}{'R0rank':>7}{'Gap@5':>8}{'R1':>5}{'R2':>5}{'R3win':>8}{'trunc':>7}")
    for c in all_cases:
        r3v = c.get("r3_vs_full_gold_score", {})
        print(
            f"  {c['question'][:32]:<34}{str(c.get('r0_gold_rank')):>7}{str(c.get('gap@5')):>8}"
            f"{str(c.get('r1_gold_rank')):>5}{str(c.get('r2_gold_rank')):>5}"
            f"{str(round(r3v.get('best_window_score', 0), 2)):>8}{str(any(c.get('was_truncated', []))):>7}"
        )

    out = OUT_DIR / f"step6_reranker_anatomy_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": all_cases,
                "rerank_failures": rerank_failures,
                "guardrail_cases": guardrail_cases,
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

"""
Step 8: Gold 重标注（40 个 exact_match 问题）

目标：把 40 个 document-level Gold 重新标注为 chunk-level Gold：
  - answerability: answerable / partially_answerable / unsupported
  - answer_bearing_chunk_ids: 该问题所有承载答案的 chunk（允许多个）
  - evidence_spans: 每个 chunk 内的答案片段（人工校对参考）
  - evidence_type: single_chunk / cross_chunk / multi_hop
  - chunking_failure: 该问题是否因 chunk 切分导致检索失败

流程：
  1. 对每个问题收集候选证据：expected_doc 的所有 chunk + 混合检索 Top20
  2. LLM answerability 标注（DeepSeek，结构化 JSON）
  3. 规则交叉验证（keyword overlap 等）
  4. 输出合并结果，人工审查后写入 tests/test_questions.json

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/step8_gold_relabel.py

产出: eval_results/step8_gold_relabel_<timestamp>.json
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
from src.retriever import Retriever  # noqa: E402

MAX_CANDIDATES = 20  # 每个问题收集的最大候选 chunk 数


def gold_filename(expected: str) -> str:
    return expected.rsplit(".", 1)[0]


def is_gold_chunk(chunk: dict, expected: str) -> bool:
    fn = chunk["metadata"].get("filename", "")
    return expected == fn or fn == gold_filename(expected)


# ── LLM Answerability 标注 ─────────────────────────────

ANNOTATION_SYSTEM_PROMPT = """\
你是一个医学 RAG 评测数据标注助手。给定一个用户问题和一组候选证据片段，判断该问题是否可由知识库回答，并找出承载答案的片段。

## 输出要求（严格 JSON）
{
  "answerability": "answerable" | "partially_answerable" | "unsupported",
  "answer_bearing_chunk_ids": ["承载答案的 chunk id 列表（可为空）"],
  "evidence_type": "single_chunk" | "cross_chunk" | "multi_hop" | "none",
  "chunking_failure": true | false,
  "reason": "判断依据（中文，50字内）"
}

## 判定规则
- answerable: 单个或多个片段共同提供了问题的完整、直接答案
- partially_answerable: 片段提供了部分信息，但不完整（如只给了流程的一部分）
- unsupported: 片段与问题无关，或只提及主题但未提供任何答案信息
- answer_bearing_chunk_ids: 只列出真正包含答案信息的片段。同一问题可以有多个（如答案同时出现在 Abstract 和 Results）
- evidence_type:
    - single_chunk: 一个片段即可完整回答
    - cross_chunk: 答案分散在多个片段，需要合并
    - multi_hop: 需要先找到中间实体再链接到答案（如"该模型的训练数据来自哪个设备？"需要两步）
- chunking_failure: 如果答案信息确实存在但被切碎/截断导致片段无法独立承载，设为 true

## 注意
- 不要因为片段不含完整信息就标 unsupported，部分信息标 partially_answerable
- 片段文本可能被截断显示，标注时依据完整片段内容
"""

ANNOTATION_USER_PROMPT = """## 用户问题
{question}

## 候选证据片段（chunk id + 内容）
{chunks}

请输出 JSON："""


def _truncate(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def annotate_with_llm(generator, question: str, candidates: list[dict]) -> dict | None:
    """调用 LLM 标注 answerability"""
    chunks_text = []
    for c in candidates:
        chunks_text.append(f"[{c['id']}]\n{_truncate(c['text'], 400)}")
    try:
        response = generator.chat(
            messages=[
                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANNOTATION_USER_PROMPT.format(question=question, chunks="\n\n".join(chunks_text)),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        print(f"    ⚠️ LLM 标注失败: {e}")
        return None

    m = re.search(r"\{[\s\S]*\}", response)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    # 规范化
    if data.get("answerability") not in ("answerable", "partially_answerable", "unsupported"):
        data["answerability"] = "unsupported"
    if data.get("evidence_type") not in ("single_chunk", "cross_chunk", "multi_hop", "none"):
        data["evidence_type"] = "none"
    data["answer_bearing_chunk_ids"] = [str(x) for x in data.get("answer_bearing_chunk_ids", [])]
    data["chunking_failure"] = bool(data.get("chunking_failure", False))
    data["reason"] = str(data.get("reason", ""))
    data["mode"] = "llm"
    return data


# ── 规则交叉验证 ─────────────────────────────────────


def rule_overlap_annotation(question: str, keywords: list[str], candidates: list[dict]) -> dict:
    """用 expected_answer_keywords 做规则判定（与 LLM 交叉验证）"""
    hits = {}
    for c in candidates:
        text = c["text"].lower()
        matched = [kw for kw in keywords if kw.lower() in text]
        if matched:
            hits[c["id"]] = matched

    best = max((len(v) for v in hits.values()), default=0)
    if best >= max(2, len(keywords) // 2):
        answerability = "answerable"
    elif best >= 1:
        answerability = "partially_answerable"
    else:
        answerability = "unsupported"

    return {
        "answerability": answerability,
        "answer_bearing_chunk_ids": list(hits.keys()),
        "keyword_hits": hits,
        "mode": "rule",
    }


def main():
    print("=" * 70)
    print("  🔬 Step 8: Gold 重标注（40 个 exact_match → chunk-level）")
    print("=" * 70, flush=True)

    provider = get_embedding_provider("local")
    provider.warmup()
    generator = create_generator()

    store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
    print(f"  📂 Milvus: {store.count()} chunks", flush=True)
    all_docs = store.get_all_documents()
    bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
    print(f"  📂 BM25: {bm25.get_total_docs()}", flush=True)

    retriever = Retriever(
        vector_store=store,
        embedding_provider=provider,
        top_k=10,
        generator=None,
        enable_rewrite=False,
        enable_reranker=False,
        bm25_backend="disk",
        bm25_index_dir="lucene_bm25_index",
    )

    questions = [q for q in json.load(open("tests/test_questions.json", encoding="utf-8")) if q.get("expected_doc")]
    print(f"  📝 有 Gold 的问题: {len(questions)}", flush=True)

    doc_chunks: dict[str, list[dict]] = {}
    for d in all_docs:
        fn = d["metadata"].get("filename", "")
        doc_chunks.setdefault(fn, []).append(d)

    results = []
    t0 = time.time()
    for idx, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_doc"]
        base = gold_filename(expected)
        kws = q.get("expected_answer_keywords", [])
        print(f"  ── [{idx}/{len(questions)}] {q['id']} {question[:40]}", flush=True)

        # ── 候选证据收集 ──
        candidates: list[dict] = []
        seen: set = set()
        # 1. expected_doc 的所有 chunk（高优先级）
        for c in doc_chunks.get(expected, []) + doc_chunks.get(base, []):
            if c["id"] not in seen:
                seen.add(c["id"])
                candidates.append(c)
        # 2. 混合检索 Top20（补充其他可能承载答案的文档）
        for c in retriever._hybrid_retrieve(question, fetch_k=20):
            if c["id"] not in seen:
                seen.add(c["id"])
                candidates.append(c)
        candidates = candidates[:MAX_CANDIDATES]

        case = {
            "id": q["id"],
            "question": question,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_doc": expected,
            "expected_answer_keywords": kws,
            "gold_chunk_count": len(doc_chunks.get(expected, [])) + len(doc_chunks.get(base, [])),
            "candidates": [
                {
                    "id": c["id"],
                    "filename": c["metadata"].get("filename", ""),
                    "is_gold": is_gold_chunk(c, expected),
                    "text_preview": _truncate(c["text"], 150),
                }
                for c in candidates
            ],
        }

        # ── LLM 标注 ──
        llm_ann = annotate_with_llm(generator, question, candidates)
        case["llm_annotation"] = llm_ann or {"answerability": "unknown", "mode": "failed"}

        # ── 规则交叉验证 ──
        rule_ann = rule_overlap_annotation(question, kws, candidates)
        case["rule_annotation"] = rule_ann

        # ── 一致性检查 ──
        if llm_ann:
            agree = llm_ann["answerability"] == rule_ann["answerability"]
            case["agree"] = agree
            flag = "✅" if agree else "⚠️ 不一致"
            print(f"      LLM={llm_ann['answerability']}  Rule={rule_ann['answerability']}  {flag}")
            print(f"      bearing: {llm_ann['answer_bearing_chunk_ids']}")
        else:
            case["agree"] = None

        results.append(case)

    elapsed = time.time() - t0

    # ── 汇总 ──
    from collections import Counter

    llm_stats = Counter(c["llm_annotation"]["answerability"] for c in results)
    rule_stats = Counter(c["rule_annotation"]["answerability"] for c in results)
    agree_count = sum(1 for c in results if c.get("agree"))
    disagree_count = sum(1 for c in results if c.get("agree") is False)

    print("\n" + "=" * 70)
    print("  📊 Step 8 重标注汇总")
    print("=" * 70)
    print(f"  LLM answerability: {dict(llm_stats)}")
    print(f"  Rule answerability: {dict(rule_stats)}")
    print(f"  LLM vs Rule 一致: {agree_count}/{len(results)} | 不一致: {disagree_count}")

    # 新标注 vs 旧 A/B/C
    try:
        step7 = json.load(open("eval_results/step7_rerank_query_oracle_20260809_151043.json", encoding="utf-8"))
        ab = step7.get("ab_quality", {})
        print("\n  📋 按旧分类的 LLM answerability:")
        for qid, old in ab.items():
            row = next((c for c in results if c["id"] == qid), None)
            if row:
                new = row["llm_annotation"]["answerability"]
                mark = (
                    "  <<< 改判"
                    if (old == "C" and new != "unsupported") or (old in ("A", "B") and new == "unsupported")
                    else ""
                )
                print(f"    {qid:<10} old={old} → new={new}{mark}")
    except FileNotFoundError:
        pass

    out = OUT_DIR / f"step8_gold_relabel_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "llm_stats": dict(llm_stats),
                "rule_stats": dict(rule_stats),
                "agree_count": agree_count,
                "disagree_count": disagree_count,
                "cases": results,
                "elapsed": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  📄 报告: {out} (耗时 {elapsed:.0f}s)")


if __name__ == "__main__":
    main()

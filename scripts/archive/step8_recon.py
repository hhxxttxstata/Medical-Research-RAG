"""
Step 8 原型 — 快速侦察 C 类问题在语料中的证据情况

对每个有 Gold 的 exact_match 问题：
  1. 收集 expected_doc 的所有 chunk（doc_chunks）
  2. 混合检索 Top20（RAG 实际能看到的候选）
  3. 输出每个 chunk 的 id / filename / 文本片段，供人工 + LLM 判断 answerability
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_embedding_provider  # noqa: E402
from src.lucene_bm25 import LuceneBM25Index  # noqa: E402
from src.milvus_store import MilvusStore  # noqa: E402
from src.retriever import Retriever  # noqa: E402

provider = get_embedding_provider("local")
provider.warmup()
store = MilvusStore(collection_name="rag_docs_c300_500", use_lite=True)
all_docs = store.get_all_documents()
bm25 = LuceneBM25Index(index_dir="lucene_bm25_index")
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

qs = json.load(open("tests/test_questions.json", encoding="utf-8"))
step7 = json.load(open("eval_results/step7_rerank_query_oracle_20260809_151043.json", encoding="utf-8"))
c_ids = [k for k, v in step7["ab_quality"].items() if v == "C"]
by_id = {q["id"]: q for q in qs}

doc_chunks = {}
for d in all_docs:
    fn = d["metadata"].get("filename", "")
    doc_chunks.setdefault(fn, []).append(d)

for qid in c_ids:
    q = by_id[qid]
    expected = q["expected_doc"]
    base = expected.rsplit(".", 1)[0]
    chunks = doc_chunks.get(expected, []) + doc_chunks.get(base, [])
    print("=" * 80)
    print(f"{qid} | {q['question']}")
    print(f"expected_doc: {expected} | chunks: {len(chunks)}")
    kws = q.get("expected_answer_keywords", [])
    print(f"keywords: {kws}")
    for c in chunks:
        hits = [kw for kw in kws if kw.lower() in c["text"].lower()]
        flag = "  <<<" if hits else ""
        print(f"  [{c['id'][-24:]}] hit={hits} | {c['text'][:110].replace(chr(10), ' ')}{flag}")
    # 检索 Top10 看哪些文档被召回
    print("  -- hybrid Top10 --")
    top10 = retriever._hybrid_retrieve(q["question"], fetch_k=10)
    for r in top10[:10]:
        fn = r["metadata"].get("filename", "")
        mark = " <== GOLD" if (fn == expected or fn == base) else ""
        print(f"    {r['score']:.4f} {fn[:46]}{mark}")
    print()

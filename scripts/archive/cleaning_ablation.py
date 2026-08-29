"""
文档清洗效果消融 — 对比「完整清洗管线」vs「跳过清洗」的检索质量

用法:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/cleaning_ablation.py

设计:
  - 同一 81 题、同一 embedding 模型、同一 numpy 向量检索
  - 变体 A: 完整管线（_sanitize_text + MarkdownConverter + CleanupPipeline）
  - 变体 B: 跳过全部清洗（PyMuPDF 原始文本直接切分入库）
  - 对比 Hit Rate / MRR / NDCG@5，量化清洗的边际贡献

产出: eval_results/cleaning_ablation_<timestamp>.json
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

import numpy as np

from eval.metrics import compute_all_metrics  # noqa: E402
from eval.test_questions import get_test_questions  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.document_processor import SmartChunker  # noqa: E402
from src.embeddings import get_embedding_provider  # noqa: E402

CHUNK_PARAMS = {"small_min": 300, "small_max": 500, "parent_min": 800, "parent_max": 2000}


def build_index(clean: bool) -> tuple[list[dict], np.ndarray]:
    """构建向量库（clean=True 走完整管线，False 走原始文本）"""
    all_chunks = []
    for file in sorted(Path("data").iterdir()):
        if file.suffix.lower() not in (".pdf", ".md", ".txt") or file.name.startswith("."):
            continue
        doc = load_document(str(file))
        if clean:
            from src.document_processor import (
                CleanupPipeline,
                MarkdownConverter,
                MetadataExtractor,
                _sanitize_text,
            )

            raw = _sanitize_text(doc.get("full_text", ""))
            pages = [{"page": p["page"], "text": _sanitize_text(p["text"])} for p in doc.get("pages", [])]
            meta = MetadataExtractor.extract(raw, file.name)
            md = MarkdownConverter.convert(pages, meta)
            c = CleanupPipeline()
            md, _, _ = c.run(md, meta)
        else:
            # 跳过全部清洗：直接用 PyMuPDF 原始文本（含页眉页脚/断词/噪声行）
            md = doc.get("full_text", "")

        chunker = SmartChunker(**CHUNK_PARAMS)
        small_chunks, _ = chunker.chunk(md, {"file_path": str(file)})
        all_chunks.extend(small_chunks)

    texts = [c["text"] for c in all_chunks]
    provider = get_embedding_provider("local")
    provider.warmup()
    emb = provider.embed(texts, prefix="passage: ")
    return all_chunks, np.array(emb, dtype=np.float32)


def run_eval(all_chunks: list[dict], emb_matrix: np.ndarray, provider, questions) -> dict:
    """81 题 numpy 向量检索评测（与 chunk_ablation 同逻辑）"""
    records = []
    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")
        q_vec = np.array(provider.embed([question], prefix="query: ")[0], dtype=np.float32)
        scores = emb_matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:10]
        sources = []
        for i in top_idx:
            c = all_chunks[i]
            sources.append(
                {
                    "id": c.get("chunk_id", ""),
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "score": round(float(scores[i]), 4),
                    "_vector_score": round(float(scores[i]), 4),
                }
            )
        expected_hit = False
        if expected and sources:
            expected_base = expected.rsplit(".", 1)[0]
            expected_hit = any(
                expected == s["metadata"].get("filename", "") or s["metadata"].get("filename", "") == expected_base
                for s in sources
            )
        records.append(
            {
                "question": question,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "expected_doc": expected,
                "expected_hit": expected_hit,
            }
        )
    metrics = compute_all_metrics(records)
    return {"metrics": metrics, "records": records, "elapsed": round(time.time() - t0, 1)}


def main():
    print("=" * 70)
    print("  🔬 文档清洗效果消融")
    print("=" * 70, flush=True)

    questions = get_test_questions()
    results = []

    for clean in [True, False]:
        label = "完整清洗管线" if clean else "跳过清洗（原始文本）"
        print(f"\n  🔬 变体: {label} ...", flush=True)
        t0 = time.time()
        all_chunks, emb = build_index(clean)
        print(f"    📊 {len(all_chunks)} chunks 向量化完成 ({time.time() - t0:.0f}s)", flush=True)
        provider = get_embedding_provider("local")
        provider.warmup()
        r = run_eval(all_chunks, emb, provider, questions)
        r["label"] = label
        results.append(r)
        m = r["metrics"]["overall"]
        print(
            f"    ✅ {label}: hit={m['hit_rate']:.1%} mrr={m['mrr']:.3f} ndcg={m['ndcg_at_5']:.3f}",
            flush=True,
        )

    print("\n" + "=" * 70)
    print("  📊 清洗效果对比")
    print("=" * 70)
    for r in results:
        m = r["metrics"]["overall"]
        print(f"  {r['label']:<20} hit={m['hit_rate']:.1%} mrr={m['mrr']:.3f} ndcg={m['ndcg_at_5']:.3f}")
    if len(results) == 2:
        m0 = results[0]["metrics"]["overall"]
        m1 = results[1]["metrics"]["overall"]
        print(
            f"\n  📈 清洗带来的提升: Hit Rate {m1['hit_rate']:.1%} → {m0['hit_rate']:.1%} "
            f"(+{m0['hit_rate'] - m1['hit_rate']:.1%}), MRR {m1['mrr']:.3f} → {m0['mrr']:.3f}"
        )

    out = OUT_DIR / f"cleaning_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

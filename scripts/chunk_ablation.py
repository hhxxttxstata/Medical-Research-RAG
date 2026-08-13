"""
chunk 阈值消融实验 — 对比不同 small/parent chunk 尺寸的检索效果（快速版）

用法:
    PYTHONIOENCODING=utf-8 python scripts/chunk_ablation.py

设计（快速版）:
  - 文档预处理（PyMuPDF→Markdown→清洗）只做一次，缓存供所有配置复用
  - 向量检索用 numpy 直接算余弦相似度，不依赖 Milvus/BM25（消融只比相对优劣）
  - embedding 模型只加载一次，跨配置复用
  - 81 题评测逻辑与 evaluate.py 一致（expected_doc 文件名匹配判定 hit）

产出: eval_results/chunk_ablation_<timestamp>.json
注意: 本实验为 vector-only 检索（控制变量），生产 hybrid 的绝对数值见 system_eval_*
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

from eval.metrics import compute_all_metrics  # noqa: E402
from eval.test_questions import get_test_questions  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.document_processor import (  # noqa: E402
    CleanupPipeline,
    MarkdownConverter,
    MetadataExtractor,
    SmartChunker,
    _sanitize_text,
)

# 文档预处理缓存：{file_path: {"markdown": str, "metadata": dict}}
# 跨进程落盘缓存（pickle），避免重复 PyMuPDF 解析（最慢环节）
_preprocess_cache: dict[str, dict] = {}
_CACHE_FILE = Path("eval_results/.preprocess_cache.pkl")


def _load_disk_cache():
    global _preprocess_cache
    if _CACHE_FILE.exists():
        try:
            import pickle

            with open(_CACHE_FILE, "rb") as f:
                _preprocess_cache = pickle.load(f)
            print(f"  📂 加载预处理缓存: {len(_preprocess_cache)} 个文档", flush=True)
        except Exception:
            _preprocess_cache = {}


def _save_disk_cache():
    try:
        import pickle

        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(_preprocess_cache, f)
    except Exception:
        pass


def preprocess_doc(file_path: str) -> dict:
    """PyMuPDF → Markdown → 清洗（与阈值无关，只做一次缓存）"""
    if file_path in _preprocess_cache:
        return _preprocess_cache[file_path]

    filename = os.path.basename(file_path)
    doc = load_document(file_path)
    raw_text = _sanitize_text(doc.get("full_text", ""))
    pages = [{"page": p["page"], "text": _sanitize_text(p["text"])} for p in doc.get("pages", [])]

    metadata = MetadataExtractor.extract(raw_text, filename)
    markdown_text = MarkdownConverter.convert(pages, metadata)
    if not markdown_text.strip() and raw_text.strip():
        markdown_text = raw_text

    cleanup = CleanupPipeline()
    markdown_text, _, _ = cleanup.run(markdown_text, metadata)

    result = {"markdown": markdown_text, "metadata": metadata}
    _preprocess_cache[file_path] = result
    _save_disk_cache()  # 增量落盘
    return result


def chunk_markdown(preprocessed: dict, params: dict, file_path: str) -> list[dict]:
    """对预处理的 Markdown 用指定阈值切分（只取 small chunks 进向量库）"""
    meta = dict(preprocessed["metadata"])
    meta["file_path"] = file_path  # SmartChunker 用 file_path 生成 filename
    chunker = SmartChunker(**params)
    small_chunks, _ = chunker.chunk(preprocessed["markdown"], meta)
    return small_chunks


def run_one_config(label: str, params: dict, provider) -> dict:
    """用指定阈值切分 → numpy 向量检索 → 81 题评测"""
    import numpy as np

    # 1. 全量切分（复用预处理缓存）
    all_chunks = []
    for file in sorted(Path("data").iterdir()):
        if file.suffix.lower() not in (".pdf", ".md", ".txt") or file.name.startswith("."):
            continue
        pre = preprocess_doc(str(file))
        sc = chunk_markdown(pre, params, str(file))
        all_chunks.extend(sc)
        print(f"    ✅ {file.name}: {len(sc)} small chunks", flush=True)

    if not all_chunks:
        return {}

    # 2. embedding（批量）
    texts = [c["text"] for c in all_chunks]
    emb = provider.embed(texts, prefix="passage: ")
    emb_matrix = np.array(emb, dtype=np.float32)  # (N, 768)
    n_chunks = len(all_chunks)
    print(f"  📊 向量化 {n_chunks} chunks → {emb_matrix.shape}", flush=True)

    # 3. 81 题评测（numpy 余弦相似度，与 evaluate.py 的 hit 判定一致）
    questions = get_test_questions()
    records = []
    t0 = time.time()
    for q in questions:
        question = q["question"]
        expected = q.get("expected_doc", "")
        q_emb = provider.embed([question], prefix="query: ")[0]
        q_vec = np.array(q_emb, dtype=np.float32)

        # 余弦相似度（已归一化则等价于内积）
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
                "num_retrieved": len(sources),
                "top_score": round(float(scores[top_idx[0]]), 4) if len(top_idx) else 0,
            }
        )

    metrics = compute_all_metrics(records)
    return {
        "label": label,
        "params": params,
        "chunk_count": n_chunks,
        "metrics": metrics,
        "records": records,
        "elapsed": round(time.time() - t0, 1),
    }


def main():
    print("=" * 70)
    print("  🔬 Chunk 阈值消融实验（快速版：numpy 向量检索）")
    print("=" * 70, flush=True)

    # 要对比的 SmartChunker 参数（small 进向量库检索，parent 供上下文）
    configs = [
        (
            "small 200-400 / parent 600-1500",
            {"small_min": 200, "small_max": 400, "parent_min": 600, "parent_max": 1500},
        ),
        (
            "small 300-500 / parent 800-2000 (当前生产)",
            {"small_min": 300, "small_max": 500, "parent_min": 800, "parent_max": 2000},
        ),
        (
            "small 400-700 / parent 1000-3000",
            {"small_min": 400, "small_max": 700, "parent_min": 1000, "parent_max": 3000},
        ),
        (
            "small 500-800 / parent 1500-3500",
            {"small_min": 500, "small_max": 800, "parent_min": 1500, "parent_max": 3500},
        ),
    ]

    # embedding 模型只加载一次
    from src.embeddings import get_embedding_provider

    _load_disk_cache()
    provider = get_embedding_provider("local")
    provider.warmup()

    results = []
    for label, params in configs:
        print(f"\n{'─' * 70}")
        print(f"  配置: {label}")
        print(f"{'─' * 70}", flush=True)
        try:
            r = run_one_config(label, params, provider)
            if not r:
                print("  ❌ 无 chunk 生成")
                continue
            results.append(r)
            m = r["metrics"]["overall"]
            print(
                f"  ✅ chunks={r['chunk_count']} hit_rate={m['hit_rate']:.0%} "
                f"mrr={m['mrr']:.3f} ndcg={m['ndcg_at_5']:.3f} "
                f"semantic={r['metrics'].get('semantic_score', 0):.4f} ({r['elapsed']}s)",
                flush=True,
            )
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback

            traceback.print_exc()

    # 汇总对比表
    print("\n" + "=" * 70)
    print("  📊 Chunk 阈值消融对比（vector-only 检索）")
    print("=" * 70)
    header = f"{'配置':<40} {'chunks':>7} {'Hit Rate':>9} {'MRR':>7} {'NDCG@5':>8} {'语义分':>8}"
    print(header)
    print("-" * 82)
    for r in results:
        m = r["metrics"]["overall"]
        print(
            f"{r['label']:<40} {r['chunk_count']:>7} {m['hit_rate']:>8.0%} {m['mrr']:>7.3f} "
            f"{m['ndcg_at_5']:>8.3f} {r['metrics'].get('semantic_score', 0):>8.4f}"
        )
    print("-" * 82)

    out = OUT_DIR / f"chunk_ablation_{TIMESTAMP}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"configs": [c[0] for c in configs], "results": results}, f, ensure_ascii=False, indent=2, default=str
        )
    print(f"\n  📄 报告: {out}")


if __name__ == "__main__":
    main()

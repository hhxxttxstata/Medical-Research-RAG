"""
finalize_holdout.py — 校验旧 16 题 gold chunk id + 回填新 14 题 gold_chunk_ids

用法（在 scripts/rebuild_index.py 完成之后）:
    HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python scripts/finalize_holdout.py

流程:
  1. 从 milvus lite 集合目录读取全部 parquet → {chunk_id: text} 映射
  2. 校验 tests/benchmark_holdout.json 现有 16 题的 gold_chunk_ids 是否都存在
  3. 读取 scripts/_new_holdout_draft.json（子代理草稿），按 evidence_sentences
     在 chunk 文本中定位真实 chunk_id 并回填
  4. 合并输出 tests/benchmark_holdout.json（30 题）——先审查报告再确认

纪律: 只读 parquet，不启动 Milvus 连接（避免与评测抢锁）。
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MILVUS_DIR = ROOT / "milvus_db" / "milvus.db" / "collections" / "rag_docs_c300_500" / "partitions" / "_default" / "data"
OLD_JSON = ROOT / "tests" / "benchmark_holdout.json"
DRAFT_JSON = ROOT / "scripts" / "_new_holdout_draft.json"


def load_chunk_map() -> dict[str, str]:
    parquets = sorted(MILVUS_DIR.glob("*.parquet"))
    if not parquets:
        print(f"❌ 未找到 parquet: {MILVUS_DIR}")
        sys.exit(1)
    mapping: dict[str, str] = {}
    for pq in parquets:
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            cid = str(row.get("chunk_id", row.get("id", "")))
            text = str(row.get("text", ""))
            if cid:
                mapping[cid] = text
    print(f"  📂 加载 {len(mapping)} chunks（{len(parquets)} 个 parquet）")
    return mapping


def norm(s: str) -> str:
    """归一化空白，便于子串匹配"""
    return re.sub(r"\s+", "", s)


def find_chunk_by_sentence(mapping: dict[str, str], sentence: str) -> list[str]:
    """在 chunk 文本中定位包含该句子的 chunk id（先整句，再退化到前 30 字片段）"""
    s = norm(sentence)
    if not s:
        return []
    hits = [cid for cid, text in mapping.items() if s in norm(text)]
    if hits:
        return hits
    # 退化：句子前 30 个字符片段
    frag = s[:30]
    if len(frag) >= 8:
        hits = [cid for cid, text in mapping.items() if frag in norm(text)]
        if hits:
            return [f"{cid} (fragment)" for cid in hits]
    return []


def main():
    mapping = load_chunk_map()

    # ── 1. 校验旧 16 题 ──
    old = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    print("\n── 旧 16 题 gold id 校验 ──")
    missing_old = 0
    for b in old["benchmark"]:
        for h in b.get("hops", []):
            for gid in h.get("gold_chunk_ids", []):
                if gid not in mapping:
                    print(f"  ❌ {b['id']} hop{h['hop']}: {gid} 不存在")
                    missing_old += 1
    if missing_old == 0:
        print("  ✅ 旧 16 题全部 gold id 存在（重建 chunk 编号与旧索引一致）")

    # ── 2. 回填新 14 题 ──
    if not DRAFT_JSON.exists():
        print("\n⚠️ 未找到草稿 scripts/_new_holdout_draft.json，跳过回填")
        return
    draft = json.loads(DRAFT_JSON.read_text(encoding="utf-8"))
    print(f"\n── 新题 gold id 回填（{len(draft['benchmark'])} 题）──")
    report = []
    for b in draft["benchmark"]:
        for h in b.get("hops", []):
            sent = (h.get("evidence_sentences") or [""])[0]
            hits = find_chunk_by_sentence(mapping, sent) if sent else []
            if hits:
                h["gold_chunk_ids"] = [x.split(" (fragment)")[0] for x in hits]
                note = "fragment" if any("fragment" in x for x in hits) else "full"
            else:
                h["gold_chunk_ids"] = []
                note = "NOT FOUND"
            report.append(f"  {b['id']} hop{h['hop']}: {note} {h['gold_chunk_ids']}")
    print("\n".join(report))

    n_missing = sum(1 for r in report if "NOT FOUND" in r)
    print(f"\n  未定位句子的 hop: {n_missing}")

    # ── 3. 合并输出（不覆盖，写 30 题版到临时文件）──
    merged = {
        "_comment": old["_comment"] + "（已扩至 30 题：16 旧 + 14 新，新题 gold id 由 finalize_holdout.py 回填）",
        "benchmark": old["benchmark"] + draft["benchmark"],
    }
    out = ROOT / "tests" / "benchmark_holdout_30.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  📄 30 题合并版（待审查）: {out}")


if __name__ == "__main__":
    main()

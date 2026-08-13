"""
Step 8: 将 gold 重标注写入 tests/test_questions.json

从 eval_results/step8_gold_relabel_<latest>.json 读取 LLM 标注，
为每个 exact_match 问题写入：
  - gold_evidence: {
      answerability, answer_bearing_chunk_ids, evidence_type, chunking_failure
    }

用法:
    PYTHONIOENCODING=utf-8 python scripts/step8_apply_gold.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RELABEL_FILES = sorted(glob.glob("eval_results/step8_gold_relabel_*.json"))
if not RELABEL_FILES:
    print("❌ 未找到 step8_gold_relabel 报告")
    sys.exit(1)
relabel_path = RELABEL_FILES[-1]
print(f"📄 使用标注报告: {relabel_path}")

relabel = json.load(open(relabel_path, encoding="utf-8"))
by_id = {c["id"]: c for c in relabel["cases"]}

qs_path = "tests/test_questions.json"
questions = json.load(open(qs_path, encoding="utf-8"))

updated = 0
for q in questions:
    ann = by_id.get(q["id"])
    if not ann or not q.get("expected_doc"):
        continue
    llm = ann["llm_annotation"]
    if llm.get("mode") == "failed" or llm.get("answerability") == "unknown":
        continue
    q["gold_evidence"] = {
        "answerability": llm["answerability"],
        "answer_bearing_chunk_ids": llm.get("answer_bearing_chunk_ids", []),
        "evidence_type": llm.get("evidence_type", "none"),
        "chunking_failure": bool(llm.get("chunking_failure", False)),
        "annotation_source": "llm",
        "annotation_timestamp": relabel.get("timestamp", ""),
    }
    updated += 1

with open(qs_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

from collections import Counter

stats = Counter(q["gold_evidence"]["answerability"] for q in questions if "gold_evidence" in q)
print(f"✅ 已更新 {updated} 题 → tests/test_questions.json")
print(f"   answerability: {dict(stats)}")

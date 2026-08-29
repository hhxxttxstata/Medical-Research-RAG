.PHONY: evaluate evaluate-quick evaluate-ablation evaluate-history audit-badcases qualify-rules qualify-full regression locate

# 一键完整评测
evaluate:
	python evaluate.py

# 快速模式（chunk=500 + top_k=3/5/8）
evaluate-quick:
	python evaluate.py --quick

# 含消融实验
evaluate-ablation:
	python evaluate.py --ablation

# 查看历史趋势
evaluate-history:
	@python -c "import json; [print(json.dumps(h,ensure_ascii=False)) for h in [json.loads(l) for l in open('eval_results/eval_history.jsonl')][-10:]]" 2>/dev/null || echo "没有历史记录"

# ── 七阶段闭环（issue #4）────────────────────────────

# ①收集：bad case 登记表 schema 审计
audit-badcases:
	python scripts/audit_bad_cases.py

# ⑤验证（规则断言，无需索引/LLM）
qualify-rules:
	python scripts/qualify.py --rules

# ⑤验证（真实评测 + Exit Criteria，需 Milvus Lite + DeepSeek API，20-40 分钟）
qualify-full:
	python scripts/qualify.py --full

# ⑦回归（dev 18 题 Agentic + 检索 56 题，对比基线；晋升前必跑）
regression:
	python scripts/regression.py

# ②定位（失败面归类，写回 bad_cases.json 的 localize 字段）
locate:
	python scripts/locate_failure.py --dry-run

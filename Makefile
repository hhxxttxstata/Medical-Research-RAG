.PHONY: evaluate evaluate-quick evaluate-ablation evaluate-history

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

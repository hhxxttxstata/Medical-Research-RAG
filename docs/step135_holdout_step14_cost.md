# Step 13.5 + Step 14 — Holdout 泛化验收 + Cost-aware Agentic Policy

> Holdout 完成日期：2026-08-12
> v2 冻结 commit：`483b2b2`（Agentic RAG v2，不再改动）
> Holdout：`tests/benchmark_holdout.json`（16 题 / 7 类，unseen）
> 报告：`eval_results/step135_holdout_20260812_142745.json`

## 一、Step 13.5：Frozen Holdout Generalization Gate

### 纪律
- holdout 16 题开发期间从未看过 failure，v2 定型后**一次性**打开验收
- 跑完不得再拿这 16 题调 v2 —— holdout 变 dev 就是评测纪律失效
- 若暴露能力缺陷：构造新 diagnostic/dev case → 开发 v2.1/v3 → 未来建新 holdout

### 结果（V0 vs v1 vs v2，16 题）

| 指标 | V0 | v1 | v2 | 验证结论 |
|---|---|---|---|---|
| Harm | — | 0 | **0** | ✅ 安全门槛 |
| False Abstain | — | 0/14 | **0/14** | ✅ 无误拒 |
| OOD Reject | — | 2/2 | **2/2** | ✅ 继续正确拒答 |
| Unnecessary Action | — | 0/14 | **0/14** | ✅ 未过度分解 |
| Policy Action Acc | — | 7/16 | **11/16** | ✅ v2 决策更准 |
| Decomp Success | — | 0 | **4** | ✅ 结构化拆解泛化 |
| Final Rescue | — | 0 | 0 | ⚠️ market=1，未救回 |
| Evidence Recall@5 | 0.812 | 0.812 | 0.812 | 持平 |
| Operational Failure | — | — | **0/16** | ✅ 无 API 故障 |

**Rescue Market 前置条件**：Answerable=14，V0 already-hit=13，V0 miss=1（ho_hard_01）。
两版均未救回 ho_hard_01 —— 该题是**新开发素材**（未来 v2.1/v3 的 dev case），
不是 holdout 调参对象。

**Step 13 因果结论迁移情况**：
- Harm=0 / False Abstain=0 / OOD 拒答 → 全部泛化 ✅
- Hop Recall / Completeness 与 v1 持平（holdout 里 V0 本身命中率高，无 rescue market）
- Policy Action Acc / Decomp Success → v2 显著优于 v1，且这是**决策质量**层面的泛化 ✅

### 冻结清单
```
Agentic RAG v2        = src/agentic_rag.py (commit 483b2b2)
dataset version       = rag_docs_c300_500 (887 chunks, milvus lite)
index snapshot        = lucene_bm25_index
policy prompt/version = POLICY_SYSTEM_PROMPT / v2 signal rules
grader version        = GRADER_SYSTEM_PROMPT (deepseek-chat)
reranker              = BAAI/bge-reranker-v2-m3
retrieval budget      = 4 (max_iterations=2 → budget=4)
dev benchmark         = tests/benchmark_multi_hop.json (18 题, step12_benchmark_20260812_000340.json)
holdout benchmark     = tests/benchmark_holdout.json (16 题, step135_holdout_20260812_142745.json)
```

## 二、Instrumentation 修正：ERROR ≠ ABSTAIN

**问题**：generation API timeout 被降级成拒答 → OOD Reject 可能虚高（"OOD 提高了"
实际只是"API 挂了"）。

**修正**（v2 冻结后唯一允许的插桩改动）：
- `generate()` API 失败 → 返回 `[OPERATIONAL_ERROR]` 前缀，**不再降级 ABSTAIN**
- 评测端单独统计 Operational Failure Rate（step135 报告：0/16）
- 终局状态区分：`FINAL_ACCEPT` / `FINAL_ABSTAIN_UNSUPPORTED` / `FINAL_ERROR_TIMEOUT` / `FINAL_ERROR_MODEL`

## 三、Step 14：Cost-aware Agentic Policy（v2.1）

### 设计来源（Ablation → Failure case → Gate）
```
Step 11:  −Grader ≈ full          → grader 很多时候信号冗余
Step 12:  bh_ood_02               → 但 reranker 0.946 高相关 ≠ 答案被支持
结论：不是删除 Grader，而是不要 Always-on Grader —— 只在 uncertainty/conflict 时调用
```

### v2.1 架构
```
Retrieve
  ↓
Cheap Evidence Signals（零 LLM）: top1 rel / hop support / completeness / lexical overlap
  ↓
Signal Gate
  ├── multi-part（多问号/对比词）→ DECOMPOSE（零成本结构信号）
  ├── 时间敏感（年份）          → UNCERTAIN → LLM grader（bh_ood_02 模式）
  ├── clearly supported (top1≥0.5 + comp≥1.0) → ACCEPT
  ├── clearly missing (top1<0.05)             → targeted RETRIEVE / ABSTAIN
  └── uncertain / conflict                     → LLM Grader → LLM Policy
```

### A/B 结果（18 dev benchmark，v2 冻结 vs v2.1）

**第一轮（初版 gate）** —— 暴露真实退化：
| 指标 | v2 | v2.1 | 问题 |
|---|---|---|---|
| OOD Reject | 2/2 | **1/2** | ❌ bh_ood_02 未拒答 |
| Final Rescue | 1 | **0** | ❌ bh_partial_01 未救回 |
| Evidence Recall@5 | 0.889 | 0.806 | ❌ |
| Grader Calls | 18 | 0 | ✅ |
| LLM Calls | 51 | 17 (-67%) | ✅ |
| Latency avg | 332s | 244s (-27%) | ✅ |

**根因**：cheap gate 在单跳 top1 高时直接 ACCEPT——
1. multi-hop 题被单跳 ACCEPT（丢 hop 证据）
2. bh_ood_02 词面有"肺栓塞/指南"重叠 → CONFLICT 检测失效 → 误 ACCEPT

**修复（保持零成本信号哲学）**：
1. multi-part 结构信号 → DECOMPOSE（比盲目 ACCEPT 更划算）
2. 时间敏感信号（问题含年份）→ UNCERTAIN → grader 裁决

**第二轮（修复版）—— 能力持平 + 成本 -94% grader / -50% LLM** ✅
| 指标 | v2 | v2.1 | 结论 |
|---|---|---|---|
| Evidence Recall@5 | 0.889 | **0.889** | ✅ 持平 |
| Hop Recall@5 / Completeness | 0.889 | **0.889** | ✅ 持平 |
| Final Rescue | 1 | **1** | ✅ bh_partial_01 保住 |
| Harm | 0 | **0** | ✅ |
| OOD Reject | 2/2 | **2/2** | ✅ bh_ood_02 grader 路径拒答 |
| False Abstain | 0/16 | **0/16** | ✅ |
| Policy Action Acc | 15/18 | **15/18** | ✅ |
| **LLM Grader Calls** | 18/18 | **1/18** | **-94%** |
| **LLM Calls/题** | 2.89 | **1.44** | **-50%** |

关键机制：
- 18 题中 17 题走 cheap_signal（grader=False），只 1 题走 LLM grader（bh_ood_02 时间敏感）
- multi-part 信号让 v2.1 在 bh_multi_01/02/03、bh_comp_01/02/03、bh_partial_01/02
  全部走 DECOMPOSE，rescue 能力与 v2 一致
- 唯一 timeout（bh_multi_03 api_error）被 observability 单独记录（timeout_v21=1），
  不计入 ABSTAIN/OOD —— ERROR ≠ ABSTAIN 语义落地

报告：`eval_results/step14_cost_ablation_20260813_001351.json`


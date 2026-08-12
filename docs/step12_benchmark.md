# Step 12 — Multi-hop / Agent Capability Benchmark

> 完成日期：2026-08-11
> 数据：[tests/benchmark_multi_hop.json](../tests/benchmark_multi_hop.json)（18 题 × 7 类）
> 指标：[eval/rescue_metrics.py](../eval/rescue_metrics.py)（Step 12 指标族）
> 评测：[scripts/step12_benchmark_eval.py](../scripts/step12_benchmark_eval.py)
> 报告：[eval_results/step12_benchmark_20260811_113622.json](../eval_results/step12_benchmark_20260811_113622.json)

## 一、为什么做这一步

现有 40 道 answerable 题 Fixed RAG 已 40/40——Agent 没有 Rescue 空间。
要证明"Agent 能解决 Fixed RAG 解决不了的复杂可回答问题"（Step 13 v2 的核心），
必须先构建一个 **Fixed RAG 真正有机会失败的数据集**。

## 二、Benchmark 设计（18 题 × 7 类）

| 类型 | 数量 | 目的 |
| --- | --- | --- |
| easy_single_hop | 3 | Guardrail：Agent 不应过度思考（Unnecessary Action Rate） |
| hard_single_hop | 3 | 检验 Retry（数值对比/多值检索） |
| multi_hop_composition | 3 | 检验 Decompose（同文档跨小节） |
| comparison | 3 | 需要两个 evidence source |
| constraint_query | 2 | 防止 topic-match 假阳性 |
| partial_evidence | 2 | 检验继续检索（证据分散） |
| unsupported_ood | 2 | 检验 Abstain |

**hop-level Gold**（与 step12.md 规格一致）：
```
Question
  ↓
Hop 1 { sub_question, gold_chunk_ids }
  ↓
Hop 2 { sub_question, gold_chunk_ids }
  ↓
Final evidence set → Final answer
```
所有 gold chunk 经索引内容核对（18 题中修正 5 处错误标注：U-Net IoU 在 small_3、
敏感度在 small_2、TensorRT 在 small_3、sPESI 灵敏度在 small_2、CTPA 征象在 small_1）。

## 三、指标族（eval/rescue_metrics.py 新增）

```
Final Answer Accuracy     答案是否含预期关键数值
Evidence Recall@K         gold chunks 命中比例
Hop Recall@K              每个 hop 至少 1 个 gold 命中
Evidence Completeness     最终证据对 hop 需求的总覆盖
Final Rescue / Harm       相对 V0 的救回/打掉（NetUtility = Rescue − Harm）
OOD Reject / False Abstain
Policy Action Accuracy    route 与预期路径宽松匹配
Decomposition Success     触发 DECOMPOSE 且最终命中
Retry Recovery            循环内 RETRIEVE 且最终命中
Unnecessary Action Rate   easy 题过度思考比例
Avg Iterations / Retrieval Calls / LLM Calls
```

## 四、结果（V0 vs Agentic v1，18 题）

| 指标 | 结果 |
| --- | --- |
| Evidence Recall@5 | 0.778（V0 同 0.778） |
| Hop Recall@5 | 0.778 |
| Completeness | 0.778 |
| **Final Rescue** | **0** |
| **Harm** | **1**（bh_partial_02：V0 命中但 Agent ABSTAIN） |
| OOD Reject | **1/2**（bh_ood_02 漏拒） |
| False Abstain | 3/16（bh_partial_01/02、bh_multi_01） |
| Decomp Success / Retry Recovery | 0 / 0 |
| Unnecessary Actions | 0/16（✅ Agent 不过度思考） |

### Failure Anatomy（为 Step 13 v2 定位改造点）

| 题 | 类型 | V0 | Agent | 失败模式 |
| --- | --- | --- | --- | --- |
| bh_partial_01 | partial | ER=0 | ER=0, ABSTAIN | **missing-hop retrieval**：gold 在候选第 14 位，两次 RETRIEVE 没救回 |
| bh_partial_02 | partial | ER=1 | ER=1, **ABSTAIN** | **evidence merge failure**：证据已命中但 grader 判不足 |
| bh_ood_02 | ood | — | **ACCEPT（漏拒）** | **time-sensitive OOD**：reranker top1=0.946 高相关但答案不存在，signal 规则误判 |
| bh_multi_01 | multi | ER=0 | ER=1, **ABSTAIN** | **decomposition 后仍拒答**：拆解检索命中但证据未合并充分 |

## 五、关键发现

### 1. Rescue = 0 的诚实解读

不是"Agent 没能力"，而是 **benchmark 里 V0 本身就命中了 16/18**——构建的数据集大部分
检索质量足够好（这正是知识库修复的成果）。真正的失败空间是 2 个 partial_evidence 题，
Agent 也没救回来。**这说明 Step 12 成功找到了 Step 13 要解决的精确问题**：
不是"再加一个 Agent"，而是 **hop-aware evidence acquisition + evidence accumulator**。

### 2. 修复过程中发现的 gold 标注错误（5 处）

这是"代码实现 ≠ benchmark 证明"的又一次实践：不是 Agent 检索失败，是我标注的
gold chunk 错了。修正后 easy_02/multi_01 恢复命中——**benchmark 自身的质量必须先于 Agent 能力验证**。

### 3. 时间敏感型 OOD 是 signal 规则盲区

bh_ood_02（"2026年ESC年会新指南"）暴露：reranker top1 高相关（主题词重叠）≠
答案存在。LLM grader 已正确识别（"未提及2026年"）但被 `top1 ≥ 0.5 → ACCEPT`
硬规则覆盖。**这是 Step 13 v2 的 Unsupported Detection 改造点**。

## 六、面试表达

1. **先制造失败，再谈能力**：40/40 的集合无法证明 Agent 价值 → 主动构建 Fixed RAG 会失败的 Stress Test（7 类 18 题）
2. **指标严谨性**：hop-level Gold + Evidence/Hop/Completeness 指标族——不是"答对了吗"，而是"证据是否按 hop 找全了"
3. **Failure Anatomy 驱动改造**：4 个失败案例精确归类为 missing-hop / merge / OOD / decomposition——Step 13 改什么由数据决定，不是拍脑袋
4. **诚实记录**：Rescue=0 如实报告，并解释"V0 本身命中率高 + 真正失败空间在 partial evidence"——评测意识比数字好看更重要

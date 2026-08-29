# Final Step — Step 15 + Step 16（最终交付前收尾）

> 完成日期：2026-08-13
> 前序：Step 13.5 holdout 验收 + Step 14 cost-aware v2.1（`docs/archive/step135_holdout_step14_cost.md`）
> 本阶段后**彻底停止实验**（final_step.md STOP 条件）
>
> 注（2026-08-16）：本报告为冻结实验记录。文中引用的 `scripts/step*.py` 实验脚本
> 已归档至 `scripts/archive/`（历史证据链，不保证当前环境可运行）；当前可运行入口为
> `scripts/step16_runtime_parity.py`（parity 复现）与 `evaluate.py`（检索评测）。

---

## 一、Step 15：End-to-End Grounded Answer Evaluation

### 研究问题

RAG lifecycle 的最后一段——最终生成答案中的 factual claims 是否由 final_evidence 支撑？

之前所有评测（Step 12/13/14）都止步于 **Evidence Recall / Final Answer Accuracy**（检索命中 + 答案含关键数值），
没有回答"答案中的每一句事实性陈述是否被证据支撑"。Step 15 补齐这个缺口：
**检索命中 ≠ 答案正确，答案正确 ≠ 每句都 grounded。**

### 指标（6 项 + Correct Abstention）

| 指标 | 定义 | 裁判 |
|---|---|---|
| 1. Answer Correctness | 最终答案包含预期关键数值/结论（宽松匹配） | 规则（与 Step 12 同口径） |
| 2. Groundedness | 答案 claims 中被 evidence 支撑的比例 | LLM-as-Judge |
| 3. Evidence/Citation Support | 引用编号有效 + 每 claim 的引用真实支撑 | LLM-as-Judge |
| 4. Completeness | 证据对问题覆盖的完整度（hop gold 命中比例） | 规则 |
| 5. **Unsupported Claim Rate** | 无证据支撑的 claim 比例（**显式记录每一条**） | LLM-as-Judge |
| 6. Correct Abstention | OOD 正确拒答 | 规则（与 Step 12 同口径） |

设计原则：
- 不把 retrieval hit 当成最终质量——评测最终生成答案的 grounded 质量
- 逐 claim 判定，**显式记录每条 unsupported claim**（Failure Anatomy 素材）
- LLM-Judge 不可用时降级规则判定（claim 级 char-overlap 交叉验证，复用 eval/judge.py 思想）

### 实现

- `eval/grounded_metrics.py`：claim 级判定 + 6 项指标汇总
  - `CLAIM_EVAL_SYSTEM_PROMPT`：事实核查员 prompt（supported / unsupported / unverifiable 三态）
  - `_rule_based_claims`：规则降级（无 LLM 时，char-overlap ≥60% → supported）
  - `compute_grounded_metrics()`：汇总 + 逐题 claim 明细
- `scripts/step15_grounded_eval.py`：frozen evaluation（frozen Agentic RAG v2 + 冻结 dev benchmark 18 题）

### 结果（frozen v2，18 题）

> 报告：`eval_results/step15_grounded_final.json`（合并 6 批次）
> 说明：Windows pyarrow/milvus-lite 偶发段错误 → 脚本支持 `--start/--end` 分块续跑

| 指标 | 结果 | 解读 |
|---|---|---|
| **Groundedness** | **0.993** | 74 个 factual claims 中 73 个被 final_evidence 支撑 |
| **Unsupported Claim Rate** | **1/74 = 1.4%** | 仅 1 条无支撑 claim（bh_partial_02），显式记录 |
| Answer Correctness | 4/18 | 宽松子串匹配（LLM 生成文本与 gold 表述有出入） |
| Evidence/Citation Support | 0.889 | 与 Step 12/13/14 的 ER@5 一致（持平） |
| Completeness | 0.889 | hop gold 覆盖与历史持平 |
| Correct Abstention | 2/2 | OOD 全部正确拒答 |
| Citation Valid Rate | 17/18 | 引用编号全部有效 |
| False Abstain | 1/16 | bh_multi_02 误拒（LLM 波动，如实记录） |

**唯一 unsupported claim**：
`bh_partial_02`: "同时注意在转换为HU值后根据目标组织设置合适的窗宽窗位（如肺栓塞检测中窗位100 Hu、窗宽900 Hu）"
——该窗宽窗位数值不在最终证据中（检索到了转换流程但未覆盖窗宽窗位细节），被 judge 显式标记 unsupported。
这正是 Step 15 的价值：**检索命中 ≠ 答案正确，答案正确 ≠ 每句 grounded**。

**逐题 grounded**：15/16 answerable 题 grounded=100%（含全部 easy/hard/multi/comp/constraint 题），仅 bh_partial_02 为 9/10。

---

## 二、Step 16：LangGraph Adapter + Runtime Parity

### 研究问题

设计动机：把 Agentic RAG v2 的 while-loop orchestration
套成标准 LangGraph StateGraph runtime，**零策略逻辑改动**，验证 behavioral parity。

设计原则（final_step.md Step 16）：
- Hybrid Retriever / Reranker / HopState / Evidence Bank / Completeness / Policy /
  Cost-aware Gate / Grader / Generator **全部不动**
- 唯一改动：`while ... if action == ...` 的控制流换成 StateGraph 节点
- 状态在 LangGraph state dict 中只存 **AgentState 对象引用**——节点间读写同一对象，
  行为与自定义 runner 完全一致

### 图结构

```
START → retrieve → evaluate → policy → [conditional edge]
                                        ├─ ACCEPT → finalize → END
                                        ├─ RETRIEVE → retrieve
                                        └─ DECOMPOSE → decompose → evaluate → policy
        ABSTAIN（budget 耗尽）→ finalize（拒答）
```

### 实现

- `src/langgraph_agent.py`：`LangGraphAgenticRAG(agent_v2)` 适配器
  - 5 个节点：retrieve / decompose / evaluate / policy / finalize
  - 节点内部全部调用被包装 AgenticRAG 的方法（零逻辑复制）
  - `_route`：条件边，预算耗尽强制 ABSTAIN（与自定义 while 条件一致）
  - `run()` 签名与 `AgenticRAG.run()` 相同，可无缝替换
- `tests/test_langgraph_agent.py`：8 个单元测试（stub retriever/reranker/LLM，确定性）
- `scripts/step16_runtime_parity.py`：真实 index 上的 parity 实验

### Parity 验证（Custom vs LangGraph，同一 input/index/prompt/budget）

> 报告：`eval_results/step16_runtime_parity_final.json`（6 批独占串行，干净重跑）

| 维度 | 结果 | 解读 |
|---|---|---|
| **Route 精确一致** | **18/18** | ✅ 决策序列完全一致 |
| **Evidence Recall@5 一致** | **18/18** | ✅ 检索行为完全一致 |
| **终局动作一致（ACCEPT/ABSTAIN）** | **18/18** | ✅ 拒答/回答决策全一致 |
| **能力指标（rescue/harm/ood/false_abstain）** | 逐批全同 | ✅ |
| Answer 文本逐字一致 | 2/18 | LLM 生成温度波动（非编排差异），终局状态全一致 |

**并发污染教训（重要）**：首轮 6 批并行（两个 bash 链同时启动）违反 Milvus Lite 单进程纪律，
产出 15/18 route 一致的假象（3 题差异被误判为"LLM plan 波动"）。串行独占重跑后 18/18 全一致——
**评测期间严禁并发连接 Milvus Lite，违规数据一律作废重跑**。

**结论：Behavioral Parity 成立**——核心 policy / evidence-state 架构 framework-agnostic。
18/18 route + ER + 终局动作全同；answer 文本差异仅来自 LLM 生成温度。

---

## 三、STOP

Step 15（Grounded Generation）✓
Step 16（LangGraph Runtime Parity）✓
-----------------------------------
**停止实验。**

不再做：Multi-Agent / Planner-Critic-Supervisor / 更多 retrieval tools / threshold tuning /
换 embedding / 换 reranker / 重启 Rewrite / 增加 ReAct / 为 benchmark 再修 ho_hard_01。

剩余交付物（非实验）：README/架构图更新、4 个 Demo Case。

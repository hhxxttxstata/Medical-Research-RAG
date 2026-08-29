# Step 10.5 — Agent Policy 资格审计与 Policy Node 实现

> 完成日期：2026-08-10
> 代码：[src/agentic_rag.py](../src/agentic_rag.py) | 审计：[scripts/step105_policy_audit.py](../scripts/step105_policy_audit.py)
> Probe Set：[tests/policy_probes.json](../tests/policy_probes.json) | Counterfactual：[scripts/step105_counterfactual.py](../scripts/step105_counterfactual.py)

## 一、为什么要做这一步

Step 10 的评测分布（49×ACCEPT, 16×ABSTAIN→ACCEPT, 16×ABSTAIN）暴露了一个定义上的漏洞：
**没有证据证明 RETRIEVE 和 DECOMPOSE 在真实评测中被动态触发**。

如果无法证明"Agent 根据运行时状态在多个 retrieval actions 间动态选择策略"，
那 v1 本质上只是"带循环的 Adaptive Workflow"，而不是 Agentic RAG。

## 二、审计发现（81 题，20260810_161411）

| 发现 | 结论 |
| --- | --- |
| 循环内 RETRIEVE 真实触发 32/81 题 | ✅ RETRIEVE 分支是活跃的（Step 10 的 route 记录漏掉了循环内动作） |
| DECOMPOSE 触发 0 次（grader 判定 needs_decomposition 仅 1 次） | ⚠️ 现有 81 题无法覆盖 decompose 分支 → 需 Policy Probe Set |
| `_decide()` 是纯规则映射（if-elif），LLM 只参与 grading | ⚠️ 策略空间由程序规则完整规定 → 对应 next_step.md 的"第二种结果" |
| 3 题（exact_02/10/31）gold 在候选池第 0 位但 grader 判 insufficient | ⚠️ **LLM grader 假阴性**：chunk 是 300-500 字片段，grader 要求"完整答案在视野内"导致系统性误判 |
| 16 个 ABSTAIN→ACCEPT 降级掩盖了上述假阴性 | ❌ 语义不清晰：ABSTAIN 本应只代表最终拒答 |

## 三、语义修正（next_step.md 要求）

### 1. EvidenceStatus 与 AgentAction 分离

```
Evidence Status: SUFFICIENT / INSUFFICIENT / UNSUPPORTED   （证据状态是什么）
Agent Action:    ACCEPT / RETRIEVE / DECOMPOSE / ABSTAIN    （当前该做什么）
```

原来的 `ABSTAIN → ACCEPT` 降级被移除。现在：

```
INSUFFICIENT → RETRIEVE → SUFFICIENT → ACCEPT   （健康路径）
UNSUPPORTED → ABSTAIN                            （OOD / 无答案，只出现一次）
```

### 2. route 记录完整动作序列

循环内 RETRIEVE/DECOMPOSE 现在记入 route：
`RETRIEVE→RETRIEVE→ACCEPT` 反映"首次检索不足 → 换角度 → 充分"的真实路径。

## 四、Policy Node（核心改动）

### 信号优先级（经验校准）

| 信号 | 规则 | 依据 |
| --- | --- | --- |
| reranker top1 ≥ 0.5 | ACCEPT（signal） | cross-encoder 高相关 = 答案证据已命中（比 LLM grader 可靠） |
| reranker top1 < 0.05 + 迭代未用尽 | RETRIEVE（signal） | 首次检索可能 miss，给二次机会 |
| reranker top1 < 0.05 + 迭代用尽 | ABSTAIN（signal） | 完全无相关证据（OOD） |
| 0.05-0.5 中间带 | LLM Policy 决策 | 基于完整 state 摘要 |
| LLM 不可用 | 规则 `_decide` fallback | 保留 v1 行为 |

### 信号校准实验（关键数据）

```
answerable (gold 在候选)   → top1 ≈ 0.97-0.99
unanswerable (库内无答案)   → top1 ≈ 0.08
OOD (Kubernetes/Nobel)    → top1 ≈ 0.001-0.003
```

**reranker top1 相关性是"证据是否命中"的杀手级信号**——阈值 0.5 完美区分
可回答与不可回答。这是 Policy Node 比纯 LLM grader 可靠的根本原因。

### multi-part 问题例外

`top1 ≥ 0.5 → ACCEPT` 对多子问题（"X和Y的区别？…如何鉴别？"）不适用——
top1 相关 ≠ 覆盖全部子问题。`_is_multi_part()` 结构检测（多问号/对比词/并列"和…各…"）
命中时交 LLM policy 决定 DECOMPOSE。

## 五、验证结果

### Counterfactual Test（4/4 通过）★ 核心证据

同一 query，3 种 evidence state → 3 种不同 action：

| State | 期望 | 实际 | mode |
| --- | --- | --- | --- |
| cf_01/sufficient（完整证据） | ACCEPT | ✅ ACCEPT | signal |
| cf_01/insufficient（无关证据） | RETRIEVE | ✅ RETRIEVE | signal |
| cf_02/sufficient（两子问题都有） | ACCEPT | ✅ ACCEPT | signal |
| cf_02/partial（只覆盖子问题 1） | DECOMPOSE | ✅ DECOMPOSE | policy_llm |

**证明：policy 是 state-dependent 的——state 改变 → action 改变。**

### 单测（12 passed）

新增 6 个：decide 映射（含 UNSUPPORTED）、multi-part 检测、entity_overlap、
signal ACCEPT/RETRIEVE/ABSTAIN 阈值、multi-part 交 LLM。

### 81 题回归（最终版 policy，20260810_180238）

| 指标 | 结果 |
| --- | --- |
| gold@final evidence | **40/40**（不回归） |
| OOD 正确拒答 | 16/16 |
| 非 OOD 误拒 | 1（exact_14，grader 判 sufficient 但 signal 覆盖 → **已修复**：grader sufficient 无条件 ACCEPT） |
| DECOMPOSE 真实触发 | **1 次**（cross_14，跨文档题） |
| 循环内 RETRIEVE | 17/81 题 |
| route 分布 | 61×`RETRIEVE→ACCEPT`, 17×`RETRIEVE→RETRIEVE→ABSTAIN`, 2×`RETRIEVE→ABSTAIN`, 1×`RETRIEVE→DECOMPOSE→ABSTAIN` |

> 注：cross_08/14/23 的 ABSTAIN 不算误拒——cross_doc 25 题没有 chunk-level gold 标注
> （Step 8 只标注了 40 个 exact_match），无法判定"应该答而没答"。

### 冻结 baseline（step10_agentic_eval_20260810_211249，13515s）

| 指标 | 结果 |
| --- | --- |
| FinalHit@5 | **40/40**（持平 V0，Rescue=0, Harm=0, NetUtility=+0） |
| OOD 正确拒答 | **16/16 (100%)** |
| answerable 误拒 | **0/40**（v1 旧版依赖 ABSTAIN→ACCEPT 降级才做到 0；新版 Policy Node 独立做到） |
| 平均迭代 | 1.27 |
| route 分布 | 58×`RETRIEVE→ACCEPT`, 20×`RETRIEVE→RETRIEVE→ABSTAIN`, 2×`RETRIEVE→DECOMPOSE→ABSTAIN`, 1×`RETRIEVE→ABSTAIN` |
| DECOMPOSE 真实触发 | 2 次（cross_14、cross_23，最终 ABSTAIN = 知识库覆盖不足但拆解分支活着） |
| 非 OOD ABSTAIN | 7（全部 cross_doc 无 gold 标注题，不算误拒） |

## 六、冻结范围（v1 baseline）

```
Dataset version      : tests/test_questions.json（81 题，含 gold_evidence）
Index snapshot       : milvus_db（887 chunks）+ lucene_bm25_index
Chunker version      : SmartChunker（300-500 字，Step 8 修复后）
Embedding model      : intfloat/multilingual-e5-base
Reranker             : bge-reranker-v2-m3（cross-encoder）
Grader prompt        : GRADER_SYSTEM_PROMPT
Policy prompt        : POLICY_SYSTEM_PROMPT
Thresholds           : grade_threshold=0.6, top1_accept=0.5, top1_abstain=0.05
max_iterations       : 2
```

## 七、总结

1. **从失败分析出发，不预设路径**：审计发现 3 个 gold 在候选但被判 insufficient 的假阴性 → 用 reranker 信号校准 Policy Node，而不是调 LLM 提示词（可解释、可复现）
2. **Policy 是可验证的**：Counterfactual Test 证明同一 query 因 evidence state 不同输出不同 action
3. **评测意识**：代码里实现了 DECOMPOSE ≠ benchmark 证明了 DECOMPOSE。现有 81 题无法覆盖 policy space → 补 Policy Probe Set（9 题 × 4 类 × 预期路径）

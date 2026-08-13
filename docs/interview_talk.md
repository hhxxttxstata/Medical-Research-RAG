# 项目讲法 —— 90 秒简历版 + 8-10 分钟深挖版

> 用途：秋招面试。两套讲法共用一条主线，深度不同。
> 核心叙事（措辞精确版）：
> **"我设计并实现了 framework-agnostic 的 Agentic RAG 核心——策略与证据状态架构由我独立设计，
> 先以自研 runner 严格评测，再用 LangGraph 标准 runtime 适配，同一 benchmark 验证行为一致。"**
> 不要说"我自研了一个 Agent 框架"。

---

## 一、90 秒简历版（电梯版）

### 一句话定位

> 一个面向肺栓塞医学问答的 Agentic RAG 系统：从"检索-生成"升级为"检索-评估-决策"的闭环，
> 证据不足就再检索、多跳问题就拆解、证据不支撑就拒答——并用 6 层评测体系验证每一步。

### 四句话展开（每条 20 秒）

**1. 检索层**：混合检索（e5 向量 + BM25 → RRF）+ Cross-Encoder 精排，81 题 Hit Rate 80%。

**2. Agentic 层**：自研 hop-aware 证据获取。LLM 每轮判断"证据够不够"：
够→生成，不够→定向再检索，多跳→结构化拆解，领域外→拒答。
最终版本 cost-aware：只用便宜信号（reranker 相关性/hop 完整性）就能判定大部分问题，
LLM grader 调用降 **94%**、每次问答 LLM 调用从 2.9 降到 1.4 次，能力零损失。

**3. 评测纪律**：16 题 unseen holdout 一次性泛化验收（Policy 决策准确率 v1 7/16 → v2 11/16）；
claim 级 grounded 评测：最终答案 74 条事实断言 73 条被证据支撑（**0.993**），
唯一 1 条无支撑断言显式记录。OOD 100% 正确拒答，0 误拒。

**4. 框架整合**：核心 policy/evidence-state 架构 framework-agnostic——
自研 runner 冻结后适配 LangGraph StateGraph，18 题同一 benchmark 验证
**route / 证据召回 / 终局决策 18/18 完全一致**。

### 一句话收尾（呼应岗位）

> 整个项目最花心思的不是"跑通"，而是**评测纪律**：holdout 一次性验收、冻结后不改、
> A/B 验证每个改动的 Pareto（能力不降 + 成本降），让每个数字都可以讲清楚来源。

---

## 二、8-10 分钟深挖版（面试问答准备）

### 开场（1 分钟）

"我做的是一个肺栓塞医学问答系统。一开始就是标准 RAG（检索+生成），
但很快发现医学问答的三个真实问题：**单轮检索证据不足**、**多跳问题拆不开**、
**LLM 高相关 ≠ 答案被支持**。所以我把系统演化成 Agentic 闭环，
每一步用一个 benchmark 验证，最终收敛成 framework-agnostic 的核心 + 两种 runtime。"

### Part 1：问题定义与检索底座（2 分钟）

**为什么先做检索**：RAG 质量上限由检索决定。
- 混合检索：`multilingual-e5-base` 向量 + Whoosh BM25 双路召回 → RRF 融合
- 消融实验（Step 1-9）证明每个组件边际贡献：rewrite 无正收益 → 冻结掉；
  reranker +0.05 Hit Rate；hybrid 是必要底座
- 检索层结果：81 题 Hit Rate 80%、MRR 0.80、NDCG@5 0.845

**面试追问准备**：
- *RRF 为什么用 k=60？* → 做过 k 消融，60 在召回与精度间最优
- *为什么砍掉 rewrite？* → Step 1-7 消融显示 rewrite 对术语精确题有损（改写丢失原文），
  且多一次 LLM 调用；规则门控保留但 LLM rewrite 关闭
- *BM25 用什么实现？* → 先用 rank-bm25 内存版，后换 Whoosh 磁盘索引（大数据量可扩展）

### Part 2：Agentic v1 —— 动态决策（1.5 分钟）

**核心**：把固定链路改成"检索 → 评估 → 决策"循环，4 种动作：
`ACCEPT`（证据够→生成）/ `RETRIEVE`（不够→换角度再检索）/ `DECOMPOSE`（多跳→拆解）/ `ABSTAIN`（领域外→拒答）。

**关键设计**：LLM grader 判证据充分性 + 规则 fallback（LLM 不可用时基于相关性阈值判定），
保证系统在 API 故障时仍能降级运行。

### Part 3：Agentic v2 —— Hop-aware Evidence Acquisition（1.5 分钟）

**动机**（failure case 驱动）：
- bh_multi_01：拆解后各子问题证据都齐了，但 grader 仍说"缺失"→ 需要 hop 级证据状态
- bh_ood_02：reranker top1=0.946 高相关，但答案根本不存在 → 高相关 ≠ 被支持

**v2 设计**：
- `AgentState / HopState`：每个 hop 记录 subquery、evidence_ids、support_status（SUPPORTED/PARTIAL/MISSING）
- `Evidence Bank`：跨轮去重累积 + 按 hop 重分配
- `Completeness`：required hop 的支持完成度 → 决定 ACCEPT 或定向补检索
- `Retrieval Budget`：预算耗尽 = ABSTAIN（防无限循环）

**结果**：16 题 unseen holdout 一次性验收——Policy 决策准确率 7/16 → **11/16**、
结构化拆解成功 0 → **4**、Harm=0、OOD 拒答 2/2、False Abstain 0。

**面试追问准备**：
- *holdout 纪律是什么？* → 16 题开发期间从未看过 failure，v2 定型后一次性验收；
  跑完不得再调 v2——holdout 变 dev 就是评测纪律失效。暴露缺陷 → 建新 dev case 开发 v3
- *为什么 ACCEPT 的新定义是 relevant AND complete AND supported 三重条件？*
  → bh_ood_02 实证：高相关 + 高完整性也可能无答案，grader 判 unsupported 时必须否决 ACCEPT

### Part 4：Agentic v2.1 —— Cost-aware Policy（1.5 分钟）

**动机**：v2 每轮都调 LLM grader+policy（18 题 51 次 LLM 调用，2.89 次/题）。
Step 11 消融发现 `-Grader ≈ full`（grader 很多时候信号冗余），但 bh_ood_02 证明信号冲突时 grader 不可删。

**设计**：Cheap Signal Gate 先行（零 LLM 成本）：
- multi-part 结构信号 → DECOMPOSE（零成本）
- 时间敏感（含年份）→ UNCERTAIN → grader（bh_ood_02 模式）
- top1 ≥ 0.5 + 词面重叠 → ACCEPT
- top1 < 0.05 → 定向 RETRIEVE / ABSTAIN
- 中间带 / 冲突 → 才调 LLM grader → LLM policy

**结果（A/B，18 题，能力完全持平）**：
| | v2 | v2.1 |
|---|---|---|
| LLM Grader Calls | 18/18 | **1/18（-94%）** |
| LLM Calls/题 | 2.89 | **1.44（-50%）** |
| Evidence Recall@5 | 0.889 | 0.889 |
| Final Rescue / Harm / OOD / False Abstain | — | 全部持平 |

**面试追问准备**：
- *第一版 gate 为什么失败？* → 直接 ACCEPT 单跳导致 multi-hop 题丢 hop 证据、OOD 漏拒；
  修复：multi-part→DECOMPOSE 优先 + 时间敏感→grader 裁决
- *便宜信号和 LLM 判定的边界怎么定？* → 只有 UNCERTAIN/冲突才调 LLM；
  判定"clearly supported / clearly missing"永远用零成本信号

### Part 5：评测体系 —— 不把 retrieval hit 当成最终质量（1.5 分钟）

**Step 15 grounded eval（claim 级）**：最终答案切分为 factual claims，
逐条判断是否被 final_evidence 支撑（LLM-as-Judge + 规则降级）：
- Groundedness **0.993**（74/74-1）
- Unsupported Claim Rate **1/74 = 1.4%**，唯一一条显式记录
  （bh_partial_02 的窗宽窗位数值不在最终证据中）
- OOD 正确拒答 2/2，Citation Valid 17/18

**面试追问准备**：
- *Groundedness 和 faithfulness 的区别？* → faithfulness 是整体忠实度评分；
  grounded 是 claim 级判定，显式列出每一条无支撑 claim（Failure Anatomy 素材）
- *LLM judge 会不会有偏？* → temperature=0，LLM 不可用时降级规则判定（char-overlap 交叉验证），
  双模式结果可交叉检查

### Part 6：LangGraph 整合 —— framework-agnostic 的证明（1.5 分钟）

**动机**：面试常见质疑"自研 runner 是不是玩具？"——与其辩解，不如证明核心可迁移。

**做法**：把 v2 的 while-loop orchestration 套成 LangGraph StateGraph（5 节点：
retrieve/decompose/evaluate/policy/finalize + 条件边），**零策略逻辑改动**——
节点内部全部调用被包装 agent 的同一方法，状态在 graph state 中只存 AgentState 引用。

**Parity 验证**（同一 input/index/prompt/budget，18 题，串行独占运行）：
| 维度 | 结果 |
|---|---|
| Route 精确一致 | **18/18** |
| Evidence Recall@5 一致 | **18/18** |
| 终局动作（ACCEPT/ABSTAIN）一致 | **18/18** |

**面试追问准备**：
- *为什么不是直接迁移到 LangGraph？* → 核心 policy 和证据状态必须 framework-agnostic
  才能严格做 ablation（自研 runner 无框架开销、可控）；冻结后适配 LangGraph 作为标准 runtime
- *Parity 实验怎么保证公平？* → 同一 retriever/reranker/generator 实例、同一 benchmark、
  同一 max_iterations；串行独占运行（并发会污染 Milvus Lite 检索结果——这也是评测纪律的一部分）
- *LangGraph 带来了什么？* → 标准可视化（get_graph）、可观测性、社区生态；行为零变化

### 收尾（30 秒）

> 这个项目我把它当作一个"有纪律的科研实验"来做：
> 每个设计决策都由 failure case 或消融实验驱动，每个数字都有报告可溯源，
> 冻结后不改、改动必须 A/B。最终收敛：能力验证（holdout 泛化 + grounded 0.993）
> + 成本优化（grader -94%）+ 工程验证（LangGraph parity 18/18）三者闭环。

---

## 三、数字速查表（背熟）

| 数字 | 含义 |
|---|---|
| 80% / 0.80 / 0.845 | 检索层 Hit Rate / MRR / NDCG@5（81 题） |
| 11/16 vs 7/16 | holdout Policy Action Acc（v2 vs v1） |
| 4 vs 0 | holdout Decomp Success（v2 vs v1） |
| 2/2 | OOD 正确拒答 |
| 0 | False Abstain / Harm |
| -94% | LLM Grader Calls（18/18 → 1/18） |
| -50% | LLM Calls/题（2.89 → 1.44） |
| 0.993 | Groundedness（74 claims 中 73 支撑） |
| 1/74 = 1.4% | Unsupported Claim Rate |
| 18/18 | LangGraph Parity（route / ER / 终局动作） |

## 四、措辞纪律

| ❌ 不要说 | ✅ 推荐说 |
|---|---|
| "我自研了一个 Agent 框架" | "我设计并实现了 framework-agnostic 的 Agentic RAG 核心" |
| "我们做了很多实验" | "每一步都由 benchmark 验证：failure case 驱动设计、消融量化边际贡献、holdout 一次性验收" |
| "检索效果提高了" | "混合检索 + reranker 消融后 Hit Rate 80%（rewrite 无正收益已冻结）" |
| "答案挺准的" | "claim 级 grounded 评测 0.993，1.4% unsupported 逐条显式记录" |

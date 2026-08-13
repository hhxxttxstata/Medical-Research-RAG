# Step 13 = Hop-aware Evidence Acquisition + Evidence Accumulator + Support-aware Policy，核心只做四件事：Hop State、Evidence Accumulator、Completeness/Support Check、Targeted Retrieval。不要加新模型，不要回头碰 Rewrite，也不要做 Multi-Agent。

## 先冻结 Step 12，不再改 benchmark

你已经：

看过所有 failure；
根据 failure 设计了 Step 13；
修过 Gold；
明确知道 bh_partial_01/02、bh_multi_01、bh_ood_02。
所以它从现在开始应该视作：

development / diagnostic benchmark

而不能再把它当最终“未知测试集”。

建议 Step 13 完成以后，再准备一份 unseen holdout，保持相同 7 类分布，但不要在开发过程中看 failure。

否则很容易出现：

Step 13
↓
精准修好了这四题
↓
18题漂亮
↓
但没有证明 policy 泛化

## Step 13 不需要“四处修”，其实可以统一成一个架构

你的四个失败：

partial_01
missing-hop retrieval

partial_02
evidence merge failure

multi_01
decomposition 后有证据但仍拒答

ood_02
高 reranker relevance ≠ answer supported
本质都指向一个问题：

当前 Agent 对“我已经拥有了哪些证据、还缺哪些证据”的表示不够显式。

所以不要做四个 case-specific patch。

改成：

Question
   ↓
Policy / Planner
   ↓
Evidence Requirements
   ↓
┌──────── Hop 1 ────────┐
│ subquery              │
│ retrieved evidence    │
│ support status        │
└───────────────────────┘

┌──────── Hop 2 ────────┐
│ subquery              │
│ retrieved evidence    │
│ support status        │
└───────────────────────┘
           ↓
   Evidence Accumulator
           ↓
   Completeness Check
           ↓
     ┌─────┴─────┐
 complete      incomplete
    ↓              ↓
 ACCEPT        missing hop
                   ↓
             targeted retrieve
                   ↓
              re-evaluate


## 13A： 先建立Hop State

建议不要让 evidence 继续只是一个 flat list。

改成类似：

HopState:
    hop_id
    subquery
    required
    evidence_ids
    evidence_score
    support_status
    retrieval_attempts
其中：

support_status:

SUPPORTED
PARTIAL
MISSING
CONTRADICTED
全局：

AgentState:
    original_query

    plan
    hops

    evidence_bank
    evidence_by_hop

    completeness

    evidence_status
    action

    retrieval_history
    iteration
    retrieval_budget

    final_evidence
最关键的就是：

evidence_by_hop
因为你现在的 bh_partial_02 和 bh_multi_01 已经证明：

“候选里有 Gold” ≠ “Agent 知道证据已经齐了”。

## 13B：Evidence Accumulator
现在不要每轮 retrieval 完就覆盖：

state["candidates"] = new_candidates
而应该：

Round 1 evidence
       +
Round 2 evidence
       +
Decomposition evidence
       ↓
dedup
       ↓
Evidence Bank
       ↓
assign evidence → hop
例如：

Hop 1:
chunk_12
chunk_14

Hop 2:
chunk_37

Unassigned:
chunk_45
然后 completeness 不再问：

“当前 top1 高不高？”

而问：

“回答这个问题需要的 evidence slots 是否都已经 supported？”

这会直接处理 partial_02。

## 13C：改变 ACCEPT 的定义
这是这轮最重要的 policy 修正。

你现在 bh_ood_02：

reranker top1 = 0.946
→ signal rule
→ ACCEPT

但是：
LLM grader 已经判断 unsupported
这个 case 已经实证说明：

Reranker relevance 不能作为 answerability 的硬证明。

所以 Step 13 里：

High reranker score
只能代表：

“这个 chunk 很相关。”

不能代表：

“这个 chunk 支持答案。”

我会把 ACCEPT 改成：

ACCEPT =
    evidence relevant
    AND
    required evidence complete
    AND
    answer support sufficient
也就是：

Relevance ≠ Support ≠ Completeness
这三个概念正式分开。

## Signal 优先级也因此应该改
目前大概是：

reranker hard signal
    ↓
LLM
    ↓
fallback
我不会继续让：

top1 > threshold
→ 强制 ACCEPT
更合理的是：

top1 极高
→ relevance prior

但如果：
grader = UNSUPPORTED
或者
completeness = false

就不能直接 ACCEPT
可以做成：

high relevance + supported
→ ACCEPT

high relevance + partial
→ RETRIEVE missing hop

high relevance + unsupported
→ RETRIEVE / ABSTAIN

low relevance
→ RETRIEVE / DECOMPOSE
你的 bh_ood_02 就是非常漂亮的反例证明。

## 13D：真正实现 targeted hop retrieval
bh_partial_01 的问题最有价值：

Gold rank = 14
两次 RETRIEVE
仍没有救回
这说明：

Repeat Retrieval ≠ Corrective Retrieval

Agent 如果只是：

original query
→ retrieve
→ 证据不足
→ original query 再 retrieve
基本没获得新信息。

应该变成：

Original:
Q

↓ identify missing hop

Need:
"X 的灵敏度是多少？"

↓ targeted subquery

retrieve(hop_query)

↓ hop-specific rerank

Evidence Bank
也就是说 RETRIEVE action 最好携带：

{
    "action": "RETRIEVE",
    "target_hop": "hop_2",
    "query": "...",
    "reason": "missing evidence for sensitivity"
}
而不是裸：

RETRIEVE
这样 Agent Action 从：

我要再搜一次
升级成：

我知道缺什么，所以我要搜索什么。

这才真正体现 agent policy。

## 13E：DECOMPOSE 不应该只是“生成几个 query”
你目前 multi_01 已经出现：

DECOMPOSE
→ retrieval 命中
→ 最终 ABSTAIN
说明 decomposition 本身成功了一半。

问题不在：

没拆出来
而在：

拆出来以后没有维护任务完成状态
所以：

DECOMPOSE
最好输出 structured plan：

[
    {
        "hop_id": 1,
        "question": "...",
        "status": "PENDING"
    },
    {
        "hop_id": 2,
        "question": "...",
        "depends_on": 1,
        "status": "PENDING"
    }
]
然后运行：

Hop1
↓
SUPPORTED

Hop2
↓
MISSING
↓
RETRIEVE

Hop2
↓
SUPPORTED

All required hops supported
↓
ACCEPT
而不是：

decompose()
↓
一堆 subqueries
↓
retrieve
↓
重新从零 grade
这就是 v2 真正值得增加的“memory”。

## 13F：ABSTAIN 也改成“证据预算耗尽后的动作”
尤其是 OOD。

你已经修正了：

ABSTAIN = final
继续保持。

我建议 v2 逻辑变成：

evidence incomplete
+
还有合理 retrieval action
+
budget > 0
↓
RETRIEVE / DECOMPOSE
只有：

support absent
+
没有新的合理 retrieval action

或者

retrieval budget exhausted
再：

ABSTAIN
这样可以降低现在：

False Abstain = 3/16
的问题。

当然对于明确 corpus scope 外的问题可以提前拒绝，不必浪费两轮 retrieval。

Step 13 成功标准也要提前冻结
不要以“18题全对”为目标。

# 建议这轮只看 5 个 exit criteria：

1. Final Rescue > 0
   首次证明 Agent 能救 Fixed RAG miss

2. Harm = 0
   bh_partial_02 必须不再被误拒

3. OOD Reject = 2/2
   修复 bh_ood_02

4. False Abstain < 3/16
   最好压到 0–1

5. Unnecessary Action Rate 仍接近 0
   不能为了 Rescue 开始过度检索
另外一个过程指标：

Hop Completeness:
0.778 → 明显上升
但 18 题样本太少，我会更关注具体计数和失败类型是否消失，而不是追小数点。

Final Rescue > 0 是这轮最重要的里程碑
因为到目前为止：

Rewrite:
Candidate Rescue > 0
Final Rescue = 0

Agentic v1:
OOD 能力 ↑
Final Rescue = 0
如果 Step 13 第一次得到：

Fixed RAG miss
↓
Agent detects missing evidence
↓
targeted hop retrieval
↓
evidence complete
↓
ACCEPT

Final Rescue > 0
你的项目故事就又上了一个台阶。

从：

Agent 知道什么时候不回答

变成：

Agent 能识别当前证据为什么不够，并自主获取缺失证据。

这是更核心的 Agentic Retrieval 能力。

# 开发时不要每次都跑 1.5 小时的全量
你已经说这次全程 CPU，一次 benchmark 大约 1.5 小时。所以 Step 13 我会把开发循环分层，避免扫参。

只需要一个顺序：

Fast policy/unit probes：119 个测试 + hop state / accumulator / conflict tests。
4 个 failure cases：partial_01 / partial_02 / multi_01 / ood_02，外加 2–3 个 easy guardrail。
18 题 frozen dev benchmark：只有结构性改动稳定后才跑。
全新 holdout：v2 定型后只用于最终验证，不根据结果反复调代码。
这一步会保护你项目最重要的“评测纪律”。

# Step 13 完成以后，不要马上做 Multi-Agent
如果 v2 成功，下一步我反而会做：

Step 14 — Cost-aware Policy

因为 Step 11 已经告诉你：

− Grader ≈ full
而 Step 12 又告诉你：

某些关键 case：
grader 信息其实正确，
只是被 hard signal 覆盖。
这两个结论组合起来非常有意思：

不是 Grader 没价值，而是不应该 always-on，也不应该被错误优先级覆盖。

以后可以变成：

clearly supported
→ cheap signal ACCEPT

clearly weak
→ RETRIEVE / ABSTAIN

signal conflict / uncertainty zone
→ LLM grader
→ Agent policy
这样 Step 14 可以同时优化：

能力
+
LLM calls
+
latency
+
cost


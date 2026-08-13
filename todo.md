# 接下来路线

## 概括
Rewrite 实验现在可以停止。

但在正式转 Agentic RAG 前，先做一次 Step 8 Gold 重标注 + Step 7 metric 清账。

然后直接进入 Agentic RAG，不要再花时间做 Adaptive Rewrite Gate。

## Step 8 — Evaluation Dataset Repair

这是目前投资回报最高的一步。

你现在有：

40 个 document-level Gold
        ↓
A answer-bearing      12
B partially-bearing   4
C doc对/chunk错       24
也就是说 60% 的 Gold 无法合法用于 chunk-level retrieval/reranker evaluation。

更关键的是，当前 16 个有效 Gold 中：

13 个 Original 已经 Top5 命中
3 个真正 miss
所以整个 Rewrite 实验实际上是在一个只有约 3 个可争取正例的数据集上寻找增益。

这不是 Rewrite 实验做得不好，而是你发现了 benchmark 本身限制了实验结论。

对 24 个 C 不要简单“重新选一个 chunk”
我建议重新标成：

question_id

expected_doc_id

answerability:
    answerable
    partially_answerable
    unsupported

answer_bearing_chunk_ids:
    [chunk_17, chunk_18]

evidence_spans:
    [...]

evidence_type:
    single_chunk
    cross_chunk
    multi_hop

chunking_failure:
    true / false
尤其允许：

多个合法 Gold chunk
不要再强制：

Question → 唯一一个 Gold chunk
医学论文里同一事实很可能同时存在于：

Abstract
Methods
Results
Table caption
Discussion
只认一个 chunk，会产生大量 false negative。


## Step 9 — 把 Step 1–7 固化

修正：

FinalHit ≠ Rescue
NetUtility 定义
baseline comparator
把 Rewrite 实验正式冻结。

之后原则上不再调 Rewrite Prompt。

## Step 10 — Agentic RAG v1

我会让你实现：

State
├─ original_query
├─ retrieval_history
├─ candidates
├─ evidence_score
├─ route
├─ iteration
└─ final_evidence
工具：

hybrid_search
decompose
evidence_grade
generate
先做最小闭环：

Retrieve
→ Grade
→ [Generate / Decompose+Retrieve / Abstain]
限制：

max_iterations = 2
Agentic RAG 的关键是动态过程和工具选择，而不是单纯把固定链路画成 Graph

### 应该直接升级成: Adaptive / Agentic Retrieval Policy

                    User Query
                         ↓
                  Retrieval Agent
                         ↓
                 Initial Retrieval
                         ↓
                  Evidence Grader
                         ↓
            ┌────────────┼─────────────┐
            ↓            ↓             ↓
        sufficient      weak        multi-hop
            ↓            ↓             ↓
         Generate     Diagnose       Decompose
                         ↓             ↓
                ┌────────┴──────┐   Subqueries
                ↓               ↓       ↓
            lexical          semantic  Retrieve
             issue            issue      ↓
                ↓               ↓       Grade
            Sparse          Dense        ↓
           expansion       expansion ────┘
                └──────┬────────┘
                       ↓
                   Retrieve
                       ↓
                    Grade
                       ↓
               sufficient?
                 /         \
               yes          no
                ↓            ↓
            Generate      Abstain

这才是你后面真正值得叫 Agentic RAG 的东西。

### Rewrite 在新 Agent 里怎么处理？

别删。降级成一个 Tool。

从原来的：

80/81 Query
→ Rewrite
变成：

Agent tools:

hybrid_search()
dense_search()
sparse_search()

expand_terminology()   ← V2
rewrite_query()        ← 保留，但低优先级
decompose_query()
rerank()
grade_evidence()
也就是说 Rewrite 从：

pipeline 的核心步骤

变成：

Agent 在特定 failure pattern 下可以选择调用的实验性工具。

这样你 Step 1–7 一点都没浪费。

### 第一版 Agentic RAG 反而不要做得太复杂

第一版我建议只给 Agent 4 种决策：

1. ACCEPT
   当前证据足够 → Generate

2. RETRIEVE
   普通检索不足 → 再检索

3. DECOMPOSE
   Multi-hop → 拆问题后检索

4. ABSTAIN
   corpus 中没有充分证据 → 拒答/说明证据不足
甚至第一版都可以暂时不把 Rewrite 接进去。

为什么？

因为你已经有实验依据：

Rewrite action → 没稳定正收益
而：

Decomposition
Unsupported detection
Evidence grading
目前反而是你数据中非常明显但尚未系统解决的问题。
# Multi-hop / Agent Capability Benchmark

目前虽然实现了复杂问题处理能力，但 benchmark 还没有真正证明：

Agent 能不能在 Fixed RAG 失败的复杂可回答问题上，通过 planning → decomposition → iterative retrieval 把答案救回来。

这应该成为 Agentic RAG v2 的核心研究问题。

# 后面的 todo 定成这个顺序

1、Step 12 — Agentic Capability / Multi-hop Benchmark
不改 Agent，补 multi-hop、partial evidence、retry、easy-control 样本；增加 hop-level Gold 和 end-to-end Answer 指标。

2、 Step 13 — Agentic RAG v2: Hop-aware Decomposition
目标必须是第一次实现：
Fixed RAG miss → Agentic Rescue > 0。
不再只证明拒答，而是证明 Agent 能解决固定 RAG 解决不了的复杂可回答问题。

3、 Step 14 — Cost-aware Policy
利用 Step 11 的 −Grader ≈ full 结论，把 LLM Grader 改成 uncertainty-triggered，而不是 always-on。

4、 Step 15 — Production / LangGraph integration
到这时候再接 LangGraph、trace、状态持久化、fallback observability，并形成最终简历版系统。

## 暂时不要做的东西

你现在尤其不要急着：

Multi-Agent
Planner Agent
Retriever Agent
Critic Agent
Medical Agent
Supervisor Agent
然后五六个 Agent 相互对话。

目前没有任何实验结果告诉你需要它们。

你的整个项目最难得的地方就是：

Architecture follows evidence。

不要到了 Agentic 阶段突然变成：

Architecture follows buzzword。

同样也先不用继续深挖 Rewrite。

Rewrite 已经冻结，是历史实验结论。除非未来 multi-hop benchmark 中出现明确：

Failure Anatomy
→ query formulation failure
再把它作为一个 Action 重新引入。

## 先不改 Agent

先制造一个 baseline 当前真正有机会失败的数据集。

现在这 40 道 answerable：

Fixed RAG 已经 40/40
这意味着 Agent 根本没有 Rescue 空间。

继续在这个集合上改 Agent，无论你怎么升级：

Hit@5 最大还是 40/40
最后只能继续展示 OOD。

所以 Step 12 要主动构建一个： ** Agentic Stress Test / Multi-hop Evaluation **

建议覆盖

| 类型                    | 目的                     |
| --------------------- | ----------------------    |
| Easy single-hop       | Guardrail，Agent 不应过度思考 |
| Hard single-hop       | 检验 Retry               |
| Multi-hop composition | 检验 Decompose           |
| Comparison            | 需要两个 evidence source   |
| Constraint query      | 防止 H1 topic-match 假阳性  |
| Partial evidence      | 检验继续检索                 |
| Unsupported/OOD       | 检验 Abstain             |

尤其是 multi-hop，不应该只标：

Question
→ Final Answer
而应该标成：

Question
   ↓
Hop 1
  question
  gold evidence chunks
   ↓
Hop 2
  question
  gold evidence chunks
   ↓
Final evidence set
   ↓
Final answer

这时传统：

Query → Hybrid → Top5
可能失败。

而：

Agent
↓
DECOMPOSE
↓
retrieve subquery A
↓
retrieve subquery B
↓
merge evidence
↓
ACCEPT
才真正有机会产生：

Rescue > 0

## Step 12 不只看 RouteAcc

你之前最大的优点就是 metric 很严谨，这里继续保持。

建议真正冻结这些指标：

Final Answer Accuracy
Evidence Recall@K
Hop Recall@K
Evidence Completeness
Final Rescue
Harm
OOD Reject
False Abstain

Policy Action Accuracy
Decomposition Success
Retry Recovery

Avg Iterations
Retrieval Calls
LLM Calls
Latency
甚至可以新增一个很好用的指标：** Unnecessary Action Rate **

例如：

Easy Query
Original evidence 已充分

Agent 却：
DECOMPOSE
→ RETRIEVE
→ RETRIEVE
→ ACCEPT
最终虽然答对了，但 policy 不好。

所以：

UnnecessaryActionRate = 不必要Agent动作数 / 总Query

这能很好展示：

Agent 不只是“能调用工具”，还知道什么时候不调用工具。

## Step 13 才正式做 Agentic RAG v2

如果 Step 12 得到比如：

30 道 multi-hop / hard queries

Fixed RAG:
18/30

Agentic v1:
20/30

Failures:
- 6 decomposition failures
- 3 missing-hop retrieval
- 1 evidence merge failure
这时才开始改代码。

而且你马上就知道改什么。

我会让 v2 从目前：

DECOMPOSE
↓
subqueries
↓
retrieve
升级成真正的 hop-aware evidence acquisition：

User Query
    ↓
Policy
    ↓
DECOMPOSE
    ↓
Structured Plan
    │
    ├── Subquery 1
    │       ↓
    │    Retrieve
    │       ↓
    │    Evidence
    │
    └── Subquery 2
            ↓
         Retrieve
            ↓
         Evidence

            ↓
     Evidence Accumulator
            ↓
       completeness?
       /          \
      no           yes
      ↓             ↓
 next missing hop  Generate

关键不是增加更多 Agent。

而是增加：

evidence memory
+
hop state
+
missing-evidence detection
State 可以逐渐变成：

original_query

plan
subqueries

completed_hops
pending_hops

evidence_by_hop
evidence_history

evidence_status

iterations
tool_budget

final_evidence
这才是下一阶段有技术含量的 Agentic RAG。

## 然后再做 Step 14：Cost-aware Agent

这个我认为会成为你另一个很漂亮的工程亮点。

因为 Step 11 已经给了一个非常明显的信号：

− Grader
≈
v1_full
这说明：

LLM Grader 很可能不是每次都值得调用。

而你又已经发现：

reranker top1：

answerable ≈ 0.97–0.99
OOD        < 0.08
那么未来完全可以设计：

             reranker score

     < low                  > high
       ↓                      ↓
 obvious weak             obvious good
       ↓                      ↓
RETRIEVE/ABSTAIN            ACCEPT

             中间灰区
                 ↓
            LLM Grader
                 ↓
             Policy
也就是：

只有 uncertainty zone 才调用昂贵模型。

于是可以比较：

Agentic v1
vs
Cost-aware Agentic v2
看：

Accuracy
OOD Reject
False Reject
LLM Calls
Latency
Token Cost
如果做到比如：

能力基本不变
LLM grading calls ↓ 60%
这个在面试里非常好讲。




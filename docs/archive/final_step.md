# 核心 Agentic RAG 研发完成，进入 Final Delivery 阶段。

# 剩余工作

| 工作                   | 优先级    | 是否需要开发新能力    |
| -------------------- | ------ | ------------ |
| Grounded Answer Eval | **必须** | 否            |
| LangGraph Adapter    | **必须** | 否，只换 runtime |
| Runtime Parity       | **必须** | 否            |
| README / 架构图         | **必须** | 否            |
| 4 个 Demo Case        | **必须** | 否            |

从研发角度看，只剩 1 次真正的模型质量实验 + 1 次框架工程验证。

# Step 15 — End-to-End Grounded Answer Evaluation
这是目前唯一比较明显的能力证据缺口。

最后做一次 frozen evaluation 即可。

建议指标不要太多，5 个足够：

Answer Correctness
Groundedness
Evidence/Citation Support
Completeness
Unsupported Claim Rate
再把 OOD：

Correct Abstention
一并算进去。

## 特别值得加入
Unsupported Claim Rate

比如最终答案包含 5 个 factual claims：

4 个有 final_evidence 支撑
1 个证据里没有
就显式记录。

这样以后你能说：

我没有把 retrieval hit 当成最终质量，而是进一步评测最终生成答案中的 factual claims 是否由 evidence 支撑。

这会补齐整个 RAG lifecycle。

# Step 16 — LangGraph Adapter + Runtime Parity
然后解决之前担心的：

现在正好做，而且现在做的风险最低。

不要迁移整个系统。

保持：

Hybrid Retriever
Reranker
HopState
Evidence Bank
Completeness
Policy
Cost-aware Gate
Grader
Generator
全部不动。

只把现有 orchestration：

while ...
    if action == ...
套成：

LangGraph StateGraph

retrieve
   ↓
accumulate
   ↓
evaluate
   ↓
policy
 /  |  |  \
A   R  D   A
然后做一个非常小的 parity test：

Custom Runner
vs
LangGraph Runner
同一组输入、同一 index、同一 prompt、同一 budget。

验证：

Answer / ER
Rescue
Harm
OOD Reject
False Abstain
Policy Route
Iterations
目标不是 LangGraph 更强，而是：

Behavioral parity。

如果通过，项目定位会非常舒服：

Agentic RAG Core
      ↑
framework-agnostic

Runtime
├── Custom Runner
└── LangGraph Adapter

# 之后彻底停止实验
我会给你一个明确 stop condition：

Step 15 Grounded Generation       ✓
Step 16 LangGraph Runtime Parity  ✓
-----------------------------------
STOP

不要再做：

Multi-Agent
Planner/Critic/Supervisor
更多 retrieval tools
继续 tuning threshold
换 embedding
换 reranker
重启 Rewrite
增加 ReAct
为了 benchmark 再修 ho_hard_01
尤其 ho_hard_01 不要碰。
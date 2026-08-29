# Step 13.5：Frozen Holdout Generalization Gate → 冻结 v2 → Step 14：Cost-aware Agentic Policy

# 接下来路线我会正式定成：

现在
│
├─ Step 13.5
│  修 ERROR ≠ ABSTAIN 语义
│  → 运行 untouched holdout
│  → V0 / v1 / v2 对比
│  → 只做一次最终泛化验证
│
├─ Freeze Agentic RAG v2
│
└─ Step 14
   Cost-aware Agentic Policy
   → uncertainty/conflict gated grader
   → latency / calls / token / timeout metrics
   → 与 frozen v2 做能力-成本 A/B

# 先修一个最后的“非策略 Bug”：API timeout ≠ ABSTAIN
你现在有：

generate API timeout
→ 降级拒答
工程上可以容错，但评测语义不应该把它算成：

ABSTAIN / unsupported
因为这是两个完全不同的事情：

Epistemic failure:
“我检索过，但 corpus 没有充分证据”
→ ABSTAIN

Operational failure:
“我有证据，但 generation API timeout”
→ ERROR / TIMEOUT
否则将来可能出现：

OOD Reject 看起来提高了
实际上只是 API 挂了。

建议最终状态至少区分：

FINAL_ACCEPT
FINAL_ABSTAIN_UNSUPPORTED
FINAL_ERROR_TIMEOUT
FINAL_ERROR_MODEL
可以允许 generation retry 1 次；仍失败就记 ERROR_TIMEOUT，不要计入 OOD Reject，也不要计入 False Abstain，单独统计 Operational Failure Rate。

这是 instrumentation 修正，不是根据 holdout 调 policy，因此应该在第一次打开 holdout 之前完成。

# 立即跑 Step 13.5：Holdout，一次性验收 v2
你已经特意留了：

tests/benchmark_holdout.json
16 题
7 类
未用于开发
现在正是使用它的时机。

而且不要只跑 v2。必须同时冻结比较：

V0 Fixed RAG
vs
Agentic v1
vs
Agentic v2
这样你才能知道 Rescue 是不是泛化。

这里最重要的不是追求“16/16”，而是验证 Step 13 的几个因果结论是否迁移：

Final Rescue > 0：前提是 holdout 里确实存在 V0 miss 的 rescue market。如果 V0 本身全命中，就不能拿它验证 Rescue。
Harm = 0：这是最重要的安全门槛之一。
False Abstain = 0 或至少不能重新出现系统性误拒。
OOD 应继续正确拒答。
Hop Recall / Completeness 应明显优于 v1。
Unnecessary Action 应继续接近 0，不能为了 Rescue 变成“什么都分解、什么都重搜”。
尤其建议先报告：

Holdout Answerable = ?
V0 already-hit      = ?
V0 miss             = ?   ← Rescue market

v2 rescued          = ?
v2 harmed           = ?
因为：  RescueRate = V0 Misses / Rescued

## 一个重要纪律：holdout 跑完不能再拿这 16 题调 v2
如果结果很好：

Freeze Agentic RAG v2
如果结果不好，也不要：

看 bh_holdout_07
→ 写规则
→ 再跑
那样 holdout 就变 dev set 了。

正确做法是：

Holdout 暴露某类能力缺陷
        ↓
构造新的 diagnostic/dev cases
        ↓
开发 v2.1 / v3
        ↓
未来再建新 holdout
你现在整个项目最大的优势就是评测纪律，继续保持。

## Holdout 通过后，Step 13 就应该正式结束
冻结：

Agentic RAG v2
commit hash
dataset version
index snapshot
policy prompt/version
grader version
reranker
retrieval budget
max iterations
benchmark dev results
benchmark holdout results
然后不要再改 v2。

到这里，你就有了三个非常清楚的 architecture baseline：

V0 Fixed Hybrid RAG
    ↓
能答普通问题
但不会判断 unsupported

Agentic v1
    ↓
动态 Retrieve / Decompose / Abstain
主要增益 = OOD safety

Agentic v2
    ↓
Hop-aware evidence acquisition
主要增益 = missing-evidence recovery
首次 Final Rescue > 0
这已经是一条很强的演化曲线。

# 正式进入 Step 14：Cost-aware Agentic Policy
这是现在最自然的下一步，而不是 Multi-Agent。

因为你的前面两组实验已经共同给了很明确的信号：

Step 11:
− Grader ≈ full
→ grader 很多时候信号冗余

Step 12:
bh_ood_02
reranker = 0.946
但答案不存在
grader 反而判断正确
→ grader 在 signal conflict 时非常重要
所以正确结论不是：

删除 Grader。

而是：

不要 Always-on Grader，只在 uncertainty / signal conflict 时调用。

架构可以变成：

Retrieve
   ↓
Cheap Evidence Signals
   │
   ├─ Completeness
   ├─ Hop Support
   ├─ Reranker
   └─ Retrieval History
   ↓
Signal confidence
   │
   ├── clearly supported
   │       ↓
   │     ACCEPT
   │
   ├── clearly missing
   │       ↓
   │ targeted RETRIEVE
   │
   └── uncertain / conflict
           ↓
       LLM Grader
           ↓
         Policy
特别是这种情况：

reranker very high
+
support/completeness says unsupported
就是典型：

CONFLICT
→ 必须调用 Grader
所以 bh_ood_02 反而成为 Step 14 policy gate 的设计依据。

## Step 14 的目标不能再是提高 Hit@5
到 v2，你已经基本把能力指标做得很好。

Step 14 的研究问题应该变成：

能否保持 v2 的 Agentic capability，同时显著减少昂贵模型调用和延迟？

比较：

Frozen Agentic v2
vs
Cost-aware Agentic v2.1
关键结果应该长这样：

                v2       cost-aware

Final Rescue     1          1
Harm             0          0
OOD Reject       ✓          ✓
False Abstain    0          0
Completeness     ✓          ≈

LLM Grader Calls 100%       ↓
Avg LLM Calls               ↓
Avg latency                 ↓
p95 latency                 ↓
Timeout rate                ↓
Token / Query               ↓
如果最后做到类似：

保持 Rescue/Harm/OOD 指标不变，同时减少 50% 的 LLM grading calls

这会是非常漂亮的工程成果。

而且和你的整个项目风格一致：

不是觉得某组件贵所以删掉

而是：
Ablation
→ 发现冗余
→ Failure case 发现不能完全删除
→ 设计 uncertainty gate
→ A/B 验证能力成本 Pareto

## Step 14 还应该顺便补 Observability
你的项目现在已经出现过两个非常值得监控的生产问题：

grader silent fallback
generation API timeout
因此 v2.1 最好开始记录：

policy_source:
  cheap_signal / llm / fallback

grader_called:
  true / false

grader_reason:
  uncertainty / conflict / forced

fallback_used:
  true / false

operational_error:
  timeout / api_error / none

retrieval_calls
grader_calls
generation_calls

iterations
latency_ms

监控要点：Policy 显式记录 fallback source 和 error type；单独监控 grader fallback rate、
model timeout rate 和 route distribution，防止能力静默退化。
​

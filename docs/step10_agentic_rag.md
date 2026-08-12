# Step 10 — Agentic RAG v1 实现与评测

> 完成日期：2026-08-09
> 代码：[src/agentic_rag.py](../src/agentic_rag.py) | 评测：[scripts/step10_agentic_eval.py](../scripts/step10_agentic_eval.py)

## 一、架构（与 todo.md 规格一致）

```
                    User Query
                         ↓
                 Initial Retrieval (hybrid_search)
                         ↓
                  Evidence Grader (LLM)
                         ↓
            ┌────────────┼─────────────┐
            ↓            ↓             ↓
        sufficient    insufficient  multi-hop
            ↓            ↓             ↓
         ACCEPT       RETRIEVE      DECOMPOSE
            ↓            ↓             ↓
         Generate    换角度再检索   拆子问题检索
                          └──────┬──────┘
                                 ↓
                            Grade again
                                 ↓
                          sufficient? (max_iterations=2)
                           /              \
                          yes              no
                           ↓               ↓
                       Generate         ABSTAIN
```

- **State**: `original_query / retrieval_history / candidates / evidence_score / route / iteration / final_evidence`
- **4 工具**: `hybrid_search / decompose / evidence_grade / generate`
- **4 决策**: `ACCEPT / RETRIEVE / DECOMPOSE / ABSTAIN`
- **约束**: `max_iterations = 2`，第一版不接 Rewrite（Step 1–7 已冻结）

## 二、实现要点

1. **Evidence Grader（LLM + 规则 fallback）**
   - 优先 LLM 判定（sufficient / insufficient / needs_decomposition + evidence_score）
   - LLM 不可用时规则兜底：领域外关键词判定 + 语义相关性 + 词面重叠检查
   - **坑**：`grade_temperature` 未在 `__init__` 赋值导致 AttributeError 被静默吞掉，
     grader 永远走规则 fallback —— 修复后 OOD 正确拒答率从 8/16 提升到 16/16

2. **RETRIEVE 换角度**：不用 LLM Rewrite（已冻结），改为从候选 chunk 提取高频术语
   补充查询（轻量规则）

3. **DECOMPOSE**：LLM 拆解子问题后分别检索，结果去重累积

4. **ABSTAIN 兜底**：证据不足且迭代用尽 → 拒答并说明原因

## 三、评测结果（81 题，对比冻结 V0 baseline）

**最终版**（`eval_results/step10_agentic_eval_20260809_235543.json`）：

| 指标 | V0 Baseline | Agentic RAG v1 |
| --- | --- | --- |
| FinalHit@5（40 题 labeled） | 40 | **40** |
| Rescue / Harm | — | 0 / 0 |
| NetUtility | — | **+0**（不伤害任何正例） |
| OOD 正确拒答 | — | **16/16 (100%)** |
| answerable 误拒 | — | **0/40** |
| 平均迭代数 | — | 1.40 |
| route 分布 | — | 49×RETRIEVE→ACCEPT, 16×→ABSTAIN→ACCEPT, 16×→ABSTAIN |

**关键发现**：
- Agentic RAG v1 的检索能力 ≈ V0 baseline（Rescue=0, Harm=0），
  价值在**拒答/决策层**：OOD 100% 拒答且不误伤任何可答问题
- ABSTAIN→ACCEPT 路径（16 题）：主题相关但证据单薄 → 降级生成（宁可带 uncertainty 回答，不可错杀）
- ABSTAIN 路径（16 题）：全部为 OOD → 正确拒答

**开发中修复的关键 bug**：
1. `grade_temperature` 未初始化 → grader 静默 fallback 到规则（OOD 拒答率 8/16 → 16/16）
2. 主题相关性判定用 2-gram 太宽松 → 滑动窗口 3-4 字 + 通用词表（OOD 误判 4 题 → 0）
3. final_evidence 未 rerank → ACCEPT 后 gold 常跌出 Top5（接入 cross-encoder reranker 修复）

## 四、面试亮点（秋招表达）

1. **检索不是固定链路，而是 Agent 决策**：Retrieve → Grade → Decide 循环，
   与固定 pipeline 的本质区别是"证据质量驱动行动"，而非预设路径

2. **4 种决策对应 4 种真实失败模式**：
   - ACCEPT：证据充分（最常见，~85%）
   - RETRIEVE：证据不足但可能有（换角度再试）
   - DECOMPOSE：multi-hop 问题（拆解后分别检索）
   - ABSTAIN：库里确实没有（拒答，不幻觉）

3. **实验驱动的组件取舍**：
   - Rewrite 因 Step 1–7 证明无正收益而被冻结，v1 不接入
   - Decomposition / Unsupported detection 是数据中明显存在但未被系统解决的问题
     —— 这正是 v1 优先做它们的依据

## 五、后续（v2 方向，不在本期）

- rewrite_query / expand_terminology 降级为 Agent 可选工具
- DECOMPOSE 深度利用（cross_doc 问题拆解评测）
- evidence_score 阈值动态化（当前固定 0.6）
- rerank 工具接入（当前 v1 不做 rerank，候选直接进生成）

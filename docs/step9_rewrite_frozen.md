# Step 9 — Rewrite 实验冻结报告

> 冻结日期：2026-08-09
> 状态：**Rewrite 实验正式冻结，后续不再调整 Rewrite Prompt**
> 下一步：Step 10 Agentic RAG v1（Rewrite 降级为 Agent 可选工具）

## 一、为什么冻结

Rewrite 实验（Step 1–7）最终结论是 **Rewrite 没有稳定的正收益**：

| 变体 | FinalHit@5 | Rescue@5 | Harm@5 | NetUtility |
| --- | --- | --- | --- | --- |
| V0 Original（baseline） | 13 | — | — | — |
| V1 Paraphrase（旧版） | 9 | 0 | 4 | **-4** |
| V2 Retriever-aware（双查询） | 13 | 0 | 1 | **-1** |
| V2 + q_rerank（约束保留） | 13 | 0 | 1 | **-1** |

但更重要的发现是：**这个结论建立在损坏的数据管道上**。

## 二、Step 8 前置修复：数据管道 4 个 bug（重要）

在做 Gold 重标注时发现知识库本身有严重缺陷，全部修复：

| # | Bug | 后果 | 修复 |
| --- | --- | --- | --- |
| 1 | `_make_small_chunks` 丢弃 <300 字短段落（`or not chunks` 只保第一段） | 中文 md 文档 85%+ 内容从未入库 | 短段落合并累积成 chunk，尾部碎块并入前块 |
| 2 | `_detect_and_flatten_columns` 把 md/txt 短行误判为双栏 PDF 从中截断 | 每行被切成两半（`conda create -n medical-ai pytho`） | 有 `#` 结构/中文章节号/表格分隔线/文件树时跳过 |
| 3 | `_normalize_headings` 把"1. Python 3.10+"列表项误加 `##` | 列表内容被当标题丢弃 | 已有 Markdown 结构时跳过；编号后带长句/冒号不判定 |
| 4 | `_to_markdown` 把"1. **数据增强**：…"误判为标题 | 关键技巧整段丢失 | 排除以 `：/:/。` 结尾的行 |

**结果**：中文文档内容覆盖率从 ~16% 提升到 ~83%（其余为标题/分隔符），
索引从 648 → 887 个唯一 chunk（此前 4256 条记录中有大量重复插入）。

> ⚠️ 这意味着 Step 1–7 的绝大部分"miss"结论是**数据管道 bug 造成的假象**，
> 不是检索/改写算法的问题。冻结前必须重跑一次基线评测（见文末）。

## 三、Step 8 重标注结论

修复数据管道后，对 40 个 document-level Gold 做了 chunk-level 重标注
（LLM answerability 标注 + 规则交叉验证）：

| 原标注 | 数量 | 重标注后 |
| --- | --- | --- |
| A（answer-bearing） | 12 | 全部 answerable |
| B（部分承载） | 4 | 全部 answerable |
| C（文档对/chunk 错） | 24 | **22 改为 answerable，1 改为 partially，1 保留** |

**24 个 C 类问题中 23 个改判为可回答** —— 这 23 个"无法合法用于评测"的问题
几乎全是 chunker 内容丢失的受害者。最终：

- answerable: 39
- partially_answerable: 1（exact_31 V/Q 机制仅部分阐述）
- unsupported: 0
- **Chunk-level Hit Rate @10 = 100%**（修复后全部 Gold chunk 可检索）

## 四、修正的指标定义（eval/rescue_metrics.py）

Step 9 把 Step 1–7 各脚本中散落的指标收敛为统一实现：

1. **FinalHit ≠ Rescue**
   - FinalHit：变体最终 Top-K 命中（含 baseline 本来命中的）
   - Rescue：baseline miss → 变体 hit（救回才叫 Rescue）
   - 之前 Step 2 的"Rescue@5 = 9"实为 FinalHit，是虚标

2. **NetUtility = Rescue − Harm**（不是 FinalHit − Harm）

3. **baseline comparator 显式声明**
   - 所有对比都相对 V0 = Original query + hybrid retrieval + rerank Top5
   - Candidate 层（池里有 Gold）与 Rerank 层（最终进 Top5）分开统计

4. **双层 Rescue 拆解**
   - CandidateRescue：V0 候选无 Gold → V1 候选池有 Gold
   - RerankRescue：候选池有 Gold → 最终 Top5
   - RerankFailure：候选池有 Gold 但 rerank 没救回来

## 五、冻结后不再做的事

- ❌ 不再调整 Rewrite Prompt（V1/V2 均已实验，无正收益）
- ❌ 不再做 Adaptive Rewrite Gate（todo.md 明确砍掉）
- ❌ 不再重新设计 Paraphrase 风格

Rewrite 作为 **Agent 的可选工具**保留（低优先级），见 Step 10。

## 六、冻结后的基准（待重跑确认）

数据管道修复后，重跑 `python evaluate.py --skip-reindex` 得到：

- Hit Rate（document-level）：100%（40/40）
- Chunk-level Hit Rate：100%（40/40）
- MRR：1.0

> 注意：这是修复后的新基线，与 Step 1–7 的历史数字（78%）不可直接比较。
> 后续所有 Agentic RAG 对比都应相对**这个新基线**。

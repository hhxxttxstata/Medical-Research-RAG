<div align="center">

# 肺栓塞医学知识库 RAG 与影像推理服务系统

**Pulmonary Embolism Medical RAG + CT Imaging Diagnosis System**

面向肺栓塞临床诊疗的「医学 RAG + Agentic 推理 + 影像诊断」系统

</div>

---

## 要解决的问题

医学知识问答与通用 RAG 有本质差异：**检索命中 ≠ 答案正确，答案正确 ≠ 每句都有证据支撑**。

1. **证据不足**：单轮检索经常漏掉关键信息（数值/机制/跨文档推理），需要"检索-评估-再检索"的闭环
2. **多跳问题**：临床问题常含多个子问题（对比、流程、多实体），单轮检索只能覆盖一部分
3. **幻觉风险**：LLM 高相关 ≠ 答案被支持（高相关性证据可能不含答案），必须显式判定"证据是否支撑答案"
4. **成本失控**：naive Agent 每轮都调 LLM grader/policy，一次问答 3+ 次 LLM 调用

## 系统演化主线

```
Fixed Hybrid RAG
      ↓
Agentic v1            Dynamic Retrieve / Decompose / Abstain
      ↓
Agentic v2            Hop-aware Evidence Acquisition（证据按 hop 追踪）
      ↓
Agentic v2.1          Cost-aware Policy（Grader -94%，LLM calls/题 -50%）
      ↓
Framework Integration Custom Runner ↔ LangGraph（18/18 behavioral parity）
```

每一步都由 benchmark 验证驱动：Step 1-9 检索链路消融 → Step 10 动态决策 → Step 12 多跳能力 → Step 13 hop 证据状态 → Step 14 成本门控 → Step 15 grounded 答案评测 → Step 16 框架整合。

## 最终架构

```
┌─────────────────────────────────────────────────────────┐
│  RAG Stack（检索底座，frozen）                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Hybrid      │→│ Reranker      │→│ Relevance Gate    │  │
│  │ Retriever   │ │ bge-reranker  │ │ + Citation Check  │  │
│  │ e5+BM25+RRF │ │ -v2-m3        │ │                   │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Agentic Core（framework-agnostic，策略与证据状态）       │
│  ┌─────────────────────────────────────────────────┐    │
│  │ AgentState / HopState / Evidence Bank            │    │
│  │ Completeness / Retrieval Budget                  │    │
│  │ Policy: ACCEPT · RETRIEVE · DECOMPOSE · ABSTAIN  │    │
│  │ Cost-aware Gate（cheap signal → grader 按需调用） │    │
│  │ Grader / Decomposer / Generator                  │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Runtime（同一 Core，两种编排）                           │
│  ┌────────────────────┐   ┌──────────────────────────┐  │
│  │ Custom Runner      │   │ LangGraph Adapter        │  │
│  │ while-loop         │   │ StateGraph 节点图         │  │
│  │ src/agentic_rag.py │   │ src/langgraph_agent.py   │  │
│  └────────────────────┘   └──────────────────────────┘  │
│          ═══════════ 18/18 Behavioral Parity ═══════════│
└─────────────────────────────────────────────────────────┘
```

设计原则：**Agentic Core 框架无关**——policy / evidence-state 架构由我独立设计，先以自研 runner 严格评测（evaluation + ablation），冻结后用 LangGraph 标准 runtime 适配，同一 regression benchmark 验证行为一致。

> **研究层 / 服务层分离（诚实交代）**：线上服务（`app.py` 的 `/chat`）跑的是上图中底部"RAG 底座"（frozen 混合检索 + 相关性门禁 + 结构化生成），保证稳定可服务；Agentic v2 / v2.1 是研究层成果——核心模块（`src/agentic_rag.py`、`src/cost_aware_agentic_rag.py`、`src/langgraph_agent.py`）完整实现并通过 dev/holdout 评测冻结。两者分离是工程决策（新策略先评测后上线），Agentic 上生产是部署层工作，非实验缺口。

## 关键结果

### 检索层（81 题，frozen）

| 指标 | 值 |
|---|---|
| Hit Rate | 80.0% |
| MRR | 0.507 |
| NDCG@5 | 0.551 |
| Refusal Accuracy | 80.2% |

> MRR / NDCG@5 按 **document-level 真实 rank** 计算：以 expected_doc 在检索结果中首次出现的实际位置计分（与 `evaluate_system.py` 的命中判定同语义），未命中计 0。

### Agentic 层（18 题 dev + 16 题 holdout，frozen）

| 指标 | v1 | v2 | v2.1 |
|---|---|---|---|
| Holdout Policy Action Acc | 7/16 | **11/16** | — |
| Holdout Decomp Success | 0 | **4** | — |
| OOD 正确拒答 | 2/2 | 2/2 | 2/2 |
| False Abstain | 0 | 0 | 0 |
| Final Rescue（dev） | — | 1 | 1 |
| **LLM Grader Calls**（dev） | — | 18/18 | **1/18（-94%）** |
| **LLM Calls/题**（dev） | — | 2.89 | **1.44（-50%）** |

> v2.1 成本指标标注（dev）：cost-aware 策略在 18 题 dev 验证（grader -94%、calls/题 -50%）；
> holdout 16 题是冻结验收集，v2 验收后未再复测 v2.1（复测属部署前验收步骤，未执行如实交代）。

### 生成层（Step 15，claim 级）

| 指标 | 值 |
|---|---|
| Groundedness | **0.993**（74 claims 中 73 被证据支撑） |
| Unsupported Claim Rate | **1/74 = 1.4%**（逐条显式记录） |
| Answer Correctness | **4/18 = 22%**（宽松子串匹配，详见下） |
| Correct Abstention | 2/2 |
| OOD 正确拒答 | 2/2 |

> **Answer Correctness 口径（诚实交代）**：该指标用宽松子串匹配把 LLM 生成文本与 gold 标准答案比对——LLM 会改写表述（同义替换、语序调整、补充上下文），子串命中率天然偏低（4/18）。它衡量的是"表述对齐度"，不是"事实正确性"；事实正确性由上一行 Groundedness（claim 级证据支撑）衡量：74 个 claims 中 73 个被证据支撑（0.993）。面试官若问"答案只有 22% 对的？"——答案是：22% 是表述对齐度，99.3% 是事实 grounded 率，两者衡量不同维度。

### 框架层（Step 16，18 题）

| 维度 | Custom vs LangGraph |
|---|---|
| Route 精确一致 | **18/18** |
| Evidence Recall@5 一致 | **18/18** |
| 终局动作（ACCEPT/ABSTAIN）一致 | **18/18** |

> 全部实验报告：`docs/step135_holdout_step14_cost.md`、`docs/final_step15_16.md`；
> 原始数据：`eval_results/`（含 holdout / cost ablation / grounded / runtime parity）

---

## 核心特性

### 🩺 CTPA 影像辅助诊断

- ResNet25d + Gated Attention MIL 模型，支持 NIfTI 格式 CTPA 影像
- 完整预处理管线：HU 裁剪 → 体部 ROI 掩膜 → 胸部切片过滤 → 2.5D slab 提取 → 多窗归一化（肺窗/纵隔窗/CTA 血管窗）
- 输出 PE 概率、风险分级（高/中/低/阴性）、slab 级注意力权重
- 以 dict 契约注入 RAG 管线，诊断数值与知识库 chunks 并行合成 LLM 回答上下文

### 📄 文档智能处理

- PyMuPDF 解析 PDF + 手写 Markdown 转换（多栏展平 / 表格识别 / 跨行断词还原）
- 手写 CleanupPipeline 数据清理管线（6 条规则 + 质量门禁评分）
- Section-aware SmartChunker：按 H1/H2/H3 构建文档树，Small-to-Big 双粒度切分

### 🛡️ 安全与稳定性

- **输入层**：规则引擎检测 5 类 15 种提示注入模式
- **输出层**：引用编号验证（validate_citations）+ 多因子相关性门禁
- **服务层**：Circuit Breaker 熔断保护 + 指数退避重试
- **数据处理层**：质量门禁评分（规则通过率 × 有效内容比 × heading 覆盖率）

## 快速开始

### 环境要求

- Python ≥ 3.10
- Docker + Docker Compose（Milvus 向量库、Redis 缓存）
- 无 GPU 要求（CPU 推理，`RERANKER_DEVICE=cpu`）

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 主 LLM（默认 deepseek-chat） |
| `REWRITE_API_KEY` / `REWRITE_BASE_URL` / `REWRITE_MODEL` | Query Rewriting 专用模型（可选） |
| `EMBEDDING_MODEL` | Embedding 模型（默认 intfloat/multilingual-e5-base） |
| `PE_MODEL_PATH` | CTPA 诊断模型权重路径（可选） |
| `API_KEY` | API 访问认证（可选，未配置则免认证） |

### 2. 启动服务

```bash
# 启动全部服务（后端 + 前端 + Milvus + Redis）
docker compose up -d --build

# 后端 API:   http://localhost:8001
# Swagger 文档: http://localhost:8001/docs
# 前端界面:   http://localhost:3000
```

### 3. 初始化知识库

```bash
# 自动处理 data/ 目录下所有 PDF/MD/TXT 文档
docker exec backend python -c "
from src.rag_pipeline import RAGPipeline
p = RAGPipeline()
p.initialize_knowledge_base(force_reindex=True)
"
```

### 4. 运行评测

```bash
# 系统级检索评测（81 题，纯检索）
docker exec backend python evaluate.py --skip-reindex

# 完整 Pipeline 评测（含 LLM 生成 + 拒答逻辑）
docker exec backend python eval/run_full_pipeline_eval.py

# Ragas 交叉验证（本地，依赖 ragas/datasets，见 requirements.txt）
python eval/run_ragas.py

# Agentic 评测（本地，无需 Docker）
HF_HUB_OFFLINE=1 python scripts/step16_runtime_parity.py --start 1 --end 18
```

> 注意：Agentic 评测需串行独占运行（Milvus Lite 单进程文件锁，并发会产生污染数据），
> 且支持 `--start/--end` 分块续跑（Windows pyarrow 偶发段错误）。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | RAG 问答（阻塞） |
| POST | `/chat/stream` | RAG 问答（SSE 流式） |
| POST | `/chat-with-ct` | RAG + CTPA 影像联合诊断（multipart: file + question） |
| POST | `/diagnosis/predict` | CTPA 影像单独诊断 |
| GET | `/diagnosis/model` | 诊断模型状态 |
| POST | `/documents/upload` | 文档入库 |
| GET | `/knowledge-base/collections` | 集合列表 |
| GET | `/logs` / `/stats` | 日志查询 / 运行统计 |
| POST | `/feedback` | 用户反馈 |

认证方式：`Authorization: Bearer <API_KEY>`（未配置 API_KEY 时免认证）

## 项目结构

```
├── app.py                    # FastAPI 入口
├── src/
│   ├── rag_pipeline.py       # RAG 主管线
│   ├── retriever.py          # 混合检索（rewrite + 双路召回 + RRF）
│   ├── reranker.py           # Cross-Encoder 重排序
│   ├── generator.py          # LLM 生成（结构化输出 + 引用验证）
│   ├── agentic_rag.py        # Agentic RAG v2（Custom Runner，frozen）
│   ├── cost_aware_agentic_rag.py  # v2.1 Cost-aware Policy
│   ├── langgraph_agent.py    # LangGraph Adapter（StateGraph runtime）
│   ├── diagnosis.py          # CTPA 诊断模型（ResNet25d + Attention MIL）
│   ├── document_processor.py # 文档处理管线
│   ├── milvus_store.py       # Milvus 向量库封装
│   ├── lucene_bm25.py        # Whoosh BM25 磁盘索引
│   └── ...
├── eval/                     # 评测体系（retrieval / rescue / grounded / ragas）
├── scripts/                  # 分步实验脚本（step9 ~ step16）
├── tests/                    # 单元测试
├── data/                     # 医学知识库文档
├── frontend/                 # Next.js 前端
└── docker-compose.yml        # 容器编排
```

## 技术栈

FastAPI · Milvus · Sentence-Transformers · PyTorch · Whoosh · Redis · LangGraph · Docker Compose · Next.js · Gradio · OpenTelemetry · PyMuPDF

## License

MIT

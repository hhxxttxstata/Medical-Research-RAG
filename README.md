<div align="center">

# 肺栓塞医学知识库 RAG 与影像推理服务系统

**Pulmonary Embolism Medical RAG + CT Imaging Diagnosis System**

面向肺栓塞临床诊疗场景的「知识库 RAG + 影像模型推理」双引擎医疗 AI 应用

</div>

---

## 项目简介

本系统面向肺栓塞（PE）医学问答、诊疗资料查询与 CTPA 影像辅助诊断场景，将**医学文献检索增强生成（RAG）**与**深度学习影像诊断**结合，实现单次请求内完成「文献佐证 + 影像判读」的联合回答。

```
用户提问 ──→ Hybrid Retrieval 检索医学知识库 ──→ 检索相关文献片段
                        │                              │
上传 CTPA 影像 ──→ ResNet25d + Attention MIL 推理 ──→ PE 概率 + 风险分级
                        │                              │
                        └──────────┬───────────────────┘
                                   ▼
                          LLM 生成联合回答
                    （文献依据 + 影像诊断 + 引用溯源）
```

## 核心特性

### 🔍 Hybrid Retrieval 多路融合检索

覆盖检索全链路的三阶段优化：

| 阶段 | 技术 | 作用 |
|------|------|------|
| 检索前 | 15 条规则门控 + LLM Query Rewriting | 口语化 query 改写为多角度医学检索查询（最多 3 条并行），领域外 query 不改写 |
| 检索中 | Dense Embedding（multilingual-e5-base）+ BM25（Whoosh 磁盘索引）双路召回 | RRF（k=60）融合排序，语义与词法信号互补 |
| 检索后 | Cross-Encoder（bge-reranker-v2-m3）精排 + 多因子相关性门禁 | 语义分 + 字符 n-gram 重叠率 + BM25 双重确认逐层校验，配合 Small-to-Big 上下文展开与引用编号验证 |

### 🩺 CTPA 影像辅助诊断

- ResNet25d + Gated Attention MIL 模型，支持 NIfTI 格式 CTPA 影像
- 完整预处理管线：HU 裁剪 → 体部 ROI 掩膜 → 胸部切片过滤 → 2.5D slab 提取 → 多窗归一化（肺窗/纵隔窗/CTA 血管窗）
- 输出 PE 概率、风险分级（高/中/低/阴性）、slab 级注意力权重
- 注意力热力图可视化（冠状位投影标注 + 高风险切片序列展示）
- 以 dict 契约注入 RAG 管线，诊断数值与知识库 chunks 并行合成 LLM 回答上下文

### 📄 文档智能处理

- Marker-pdf 将 PDF 转为结构化 Markdown（OCR / 表格 / 公式 / 多栏还原）
- 手写 CleanupPipeline 数据清理管线（6 条规则 + 质量门禁评分）
- Section-aware SmartChunker：按 H1/H2/H3 构建文档树，Small-to-Big 双粒度切分

### 🛡️ 安全与稳定性

- **输入层**：规则引擎检测 5 类 15 种提示注入模式
- **输出层**：引用编号验证（validate_citations）+ 多因子相关性门禁
- **服务层**：Circuit Breaker 熔断保护 + 指数退避重试
- **数据处理层**：质量门禁评分（规则通过率 × 有效内容比 × heading 覆盖率）

### 📊 评测体系

- 81 道医学 QA（exact_match / cross_doc / out_of_knowledge，easy / medium / hard）
- 指标：Hit Rate、MRR、NDCG@5、Refusal Accuracy、语义相似度、Passage Diversity
- 消融实验对比 rewrite / reranker / hybrid 各组件的边际贡献
- 评测历史自动归档，支持跨版本回归对比
- Ragas 交叉验证（Faithfulness / Answer Relevancy / Context Precision / Context Recall）

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

# Ragas 交叉验证
docker exec backend python eval/run_ragas.py
```

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
│   ├── diagnosis.py          # CTPA 诊断模型（ResNet25d + Attention MIL）
│   ├── document_processor.py # 文档处理管线（Marker + CleanupPipeline + SmartChunker）
│   ├── milvus_store.py       # Milvus 向量库封装
│   ├── lucene_bm25.py        # Whoosh BM25 磁盘索引
│   ├── prompt_injection.py   # 提示注入检测
│   └── ...
├── eval/                     # 评测体系
│   ├── run_full_pipeline_eval.py
│   ├── run_ragas.py
│   ├── metrics.py
│   └── test_questions.py
├── tests/                    # 单元测试（234 个）
├── data/                     # 医学知识库文档
├── frontend/                 # Next.js 前端
├── gradio_app.py             # Gradio 演示前端
└── docker-compose.yml        # 容器编排
```

## 评测结果

| 指标 | 值 |
|------|-----|
| Hit Rate | 80.0% |
| MRR | 0.8000 |
| NDCG@5 | 0.8451 |
| Semantic Score | 0.8618 |
| Passage Diversity | 4.26 docs/query |
| Refusal Accuracy | 80.2%（端到端） |
| 平均端到端响应 | 11.6s/题 |

> 评测环境：CPU-only，Milvus Standalone，`multilingual-e5-base` 768d

## 技术栈

FastAPI · Milvus · Sentence-Transformers · PyTorch · Whoosh · Redis · Docker Compose · Next.js · Gradio · OpenTelemetry · Marker-pdf

## License

MIT

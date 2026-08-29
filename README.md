<div align="center">

# 医疗科研知识问答 RAG

**Medical Research Knowledge Q&A RAG**

面向医疗科研文献的知识问答系统：混合检索 + Agentic 多跳检索 + 证据门禁，让每个回答都有文献支撑。

> **定位声明**：本系统仅用于学术科研辅助（文献检索问答、写作规范咨询），**不提供任何临床诊断建议**；域外问题（含诊断类）会明确拒答。

</div>

---

## 核心特性

- **📚 中英文文献处理**：PDF / Markdown 自动解析、章节感知分块
- **🔎 混合检索**：向量（e5）+ BM25 融合 + RRF 排序，可接重排序（Cross-Encoder）
- **✅ 证据门禁**：答案必须引用知识库原文，无证据支撑的表述会被拦截
- **🤖 Agentic 多跳检索**：证据不足时自动拆解问题、定向补检，直至证据充分或明确拒答
- **💰 成本感知门控**：仅在必要时调用 LLM 评审，显著降低问答成本
- **🛡️ 安全拒答**：域外/诊断类问题明确拒答；提示注入检测 + 熔断重试

## 快速开始

### 环境要求

- Python ≥ 3.10
- Docker + Docker Compose（Milvus 向量库、Redis 缓存）
- 无需 GPU

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 2. 启动服务

```bash
docker compose up -d --build
```

| 服务 | 地址 |
|------|------|
| 后端 API | http://localhost:8001 |
| Swagger 文档 | http://localhost:8001/docs |
| 前端界面 | http://localhost:3000 |

### 3. 初始化知识库

```bash
docker exec backend python -c "from src.rag_pipeline import RAGPipeline; RAGPipeline().initialize_knowledge_base(force_reindex=True)"
```

### 4. 运行评测（可选）

```bash
python evaluate.py --skip-reindex
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | RAG 问答 |
| POST | `/chat/stream` | 流式问答（SSE） |
| POST | `/documents/upload` | 文档入库 |
| GET | `/knowledge-base/collections` · `/tags` | 知识库集合 / 知识域 |
| GET | `/logs` · `/stats` | 日志 / 统计 |
| POST | `/feedback` | 用户反馈 |

**示例**：

```bash
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"question": "急性与慢性肺栓塞在 CTPA 影像上如何鉴别？"}'
```

## 评测结果（摘要）

| 指标 | 结果 |
|------|------|
| 检索 Hit Rate / MRR | 100%（27/27）/ 0.938 |
| 答案证据支撑率（Groundedness） | 0.993 |
| 域外问题拒答 / 库内误拒 | 16/16 正确拒答，0 误拒 |

> 详细实验记录见 `docs/` 目录。

## 目录结构

```
├── app.py              # FastAPI 入口
├── src/                # 核心代码：混合检索、重排序、生成、Agentic 策略与编排
├── data/               # 知识库：肺栓塞科研文献（PDF + 笔记）
├── eval/ · tests/      # 评测脚本与测试（检索 / 生成 / 拒答）
├── scripts/            # 实验与辅助脚本
├── docs/               # 技术文档
└── frontend/           # Next.js 前端
```

## 技术栈

FastAPI · Milvus · Sentence-Transformers · Whoosh BM25 · LangGraph · Redis · Docker Compose · Next.js

## License

MIT

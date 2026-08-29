# 🐳 肺栓塞科研文献 RAG 系统 Docker 部署指南

## 环境要求

| 项目 | 要求 |
|------|------|
| Docker | 24.0+（含 docker compose） |
| 系统 | Windows（WSL2）、Linux、macOS |
| 磁盘 | 至少 **5GB** 可用空间（含模型缓存） |
| 内存 | 至少 **8GB**（推荐 16GB） |
| GPU（可选） | NVIDIA GPU + CUDA（开启 reranker 时加速，CPU 亦可运行） |

## 快速开始

### 1. 配置 API Key

编辑 `.env` 文件：

```bash
# DeepSeek（国内推荐）
DEEPSEEK_API_KEY=sk-your-key
```

### 2. 启动

```bash
# 构建并启动所有服务（后端 + 前端 + Milvus + Redis）
docker compose up -d --build

# 查看日志
docker compose logs -f

# 确认服务已就绪
docker compose ps
```

### 3. 初始化知识库

```bash
# 自动处理 data/ 下所有文档（pe_literature + writing_guidelines）
docker exec backend python -c "
from src.rag_pipeline import RAGPipeline
p = RAGPipeline()
p.initialize_knowledge_base(force_reindex=True)
"
```

### 4. 访问

| 服务 | 地址 |
|------|------|
| API 服务 | http://localhost:8001 |
| Swagger 文档 | http://localhost:8001/docs |
| Next.js 前端 | http://localhost:3000 |

## 常用命令

```bash
# 查看实时日志
docker compose logs -f

# 查看某个服务的日志
docker compose logs backend
docker compose logs frontend

# 重启服务（修改代码后）
docker compose restart backend

# 停止
docker compose down

# 停止并清除数据卷（重建知识库时使用）
docker compose down -v

# 仅启动后端（无前端页面）
docker compose up -d backend
```

## 架构说明

```
pe-rag-system (docker compose)
 ├── backend   (FastAPI: app.py + src/)     → port 8000（宿主 8001）
 │    ├── RAG 管线（混合检索 + 相关性门禁）
 │    ├── Agentic 服务（LangGraph + cost-aware，/query）
 │    ├── 文档库 (data/：pe_literature + writing_guidelines)
 │    └── 日志 (logs/)
 ├── frontend  (Next.js)                    → port 3000
 ├── milvus    (Milvus 向量库 + etcd + minio) → port 19530
 └── redis     (缓存，可选降级内存)           → port 6379
```

## 数据持久化

容器使用 bind mount 挂载宿主目录，数据持久在宿主机：

| 宿主机路径 | 容器内路径 | 用途 |
|-----------|-----------|------|
| `./data/` | `/app/data/` | 文档库（知识域子目录） |
| `./logs/` | `/app/logs/` | 请求日志 |
| `./eval_results/` | `/app/eval_results/` | 评测结果 |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | 模型缓存（只读复用） |

向量数据存储在 Docker 卷中（`docker compose down -v` 会清空，需重建索引）。

## GPU 支持（可选）

如需要使用 GPU 加速 reranker（默认 CPU 已可用）：

1. **Windows WSL2**: 确保 Docker Desktop 设置中启用 WSL2 后端
2. **Linux**: 安装 `nvidia-container-toolkit`
3. 在 `docker-compose.yml` 中为 backend 添加 GPU 配置并设 `RERANKER_ENABLED=true`

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## 运行评测（容器内）

```bash
# 系统级检索评测（57 题，纯检索，无需 LLM）
docker exec backend python evaluate.py --skip-reindex

# 完整 Pipeline 评测（含 LLM 生成）
docker exec backend python eval/run_full_pipeline_eval.py

# Agentic parity 评测（串行独占）
docker exec backend python scripts/step16_runtime_parity.py --start 1 --end 18
```

## 本地免 Docker 模式

无需 Docker 时可用 Milvus Lite（复用 `milvus_db/` 本地文件库）：

```bash
# Windows
start-local.bat
# 或手动
MILVUS_LITE=true python run.py
```

注意：Milvus Lite 单进程独占（文件锁），评测与 API 服务不可同时运行；
Windows 下偶发 gRPC/段错误为 Milvus Lite 已知问题，可切换到 Docker 模式规避。

## 健康检查

```bash
curl http://localhost:8001/health
```

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `backend` 一直 restarting | 启动超时 / API Key 未配置 | 检查 `.env` 文件，检查 `docker compose logs backend` |
| `frontend` 连接不上 backend | backend 未就绪 | 等待 backend healthcheck 通过 |
| 模型加载慢 | 首次需要下载 Embedding 模型 | 首次启动约 2-5 分钟，后续会命中缓存 |
| Milvus Lite 评测崩溃 | Windows 偶发 gRPC/段错误 | 使用 Docker 模式（`MILVUS_LITE=false`） |
| Docker daemon 未运行 | Docker Desktop 未启动 | 启动 Docker Desktop，或在终端执行 `start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"` |

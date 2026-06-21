# 🐳 RAG 系统 Docker 部署指南

## 环境要求

| 项目 | 要求 |
|------|------|
| Docker | 24.0+（含 docker compose） |
| 系统 | Windows（WSL2）、Linux、macOS |
| 磁盘 | 至少 **10GB** 可用空间（含模型缓存） |
| 内存 | 至少 **8GB**（推荐 16GB） |
| GPU（可选） | NVIDIA GPU + CUDA 12.4（用于诊断推理加速） |

## 快速开始

### 1. 配置 API Key

编辑 `.env` 文件，至少配置一种 LLM：

```bash
# DeepSeek（国内推荐）
DEEPSEEK_API_KEY=sk-your-key
```

### 2. 准备模型权重（可选）

如需要使用肺栓塞 AI 诊断功能，将模型权重文件放入：

```
models/
  └── best.pth       # 肺栓塞诊断模型权重（~136MB）
```

### 3. 启动

```bash
# 构建并启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f

# 确认服务已就绪
docker compose ps
```

### 4. 访问

| 服务 | 地址 |
|------|------|
| API 服务 | http://localhost:8000 |
| Swagger 文档 | http://localhost:8000/docs |
| Gradio 前端 | http://localhost:7860 |

## 常用命令

```bash
# 查看实时日志
docker compose logs -f

# 查看某个服务的日志
docker compose logs -f backend

# 重启服务（修改代码后）
docker compose restart backend

# 重新构建镜像（修改 Dockerfile 或依赖后）
docker compose build --no-cache

# 停止服务
docker compose down

# 停止并删除 volumes（会清空 data/chroma_db/logs）
docker compose down -v

# 仅启动后端（无前端页面）
docker compose up -d backend
```

## 架构说明

```
rag-system (docker compose)
 ├── backend  (rag-backend)
 │    ├── FastAPI 服务 (app.py)      → port 8000
 │    ├── RAG 管道 (src/)
 │    ├── Agent 引擎 (src/agent.py)
 │    ├── 诊断模型 (models/best.pth)
 │    ├── ChromaDB (chroma_db/)
 │    ├── 文档库 (data/)
 │    └── 日志 (logs/)
 │
 └── frontend (rag-frontend)
      └── Gradio UI (gradio_app.py) → port 7860
```

## 数据持久化

容器使用 bind mount 挂载宿主目录，数据持久在宿主机：

| 宿主机路径 | 容器内路径 | 用途 |
|-----------|-----------|------|
| `./models/` | `/app/models/` | 诊断模型权重（只读） |
| `./data/` | `/app/data/` | 文档库 |
| `./chroma_db/` | `/app/chroma_db/` | 向量数据库 |
| `./logs/` | `/app/logs/` | 请求日志 |

## GPU 支持

如需要使用 GPU 加速：

1. **Windows WSL2**: 确保 Docker Desktop 设置中启用 WSL2 后端
2. **Linux**: 安装 `nvidia-container-toolkit`

然后在 `docker-compose.yml` 中找到 GPU 配置区块，取消注释：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

验证 GPU 是否可用：

```bash
docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"
```

## 健康检查

```bash
# 检查后端服务状态
curl http://localhost:8000/health

# 前端状态（浏览器打开）
http://localhost:7860
```

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `backend` 一直 restarting | 启动超时 / API Key 未配置 | 检查 `.env` 文件，检查 `docker compose logs backend` |
| `frontend` 连接不上 backend | backend 未就绪 | 等待 backend healthcheck 通过 |
| 模型加载慢 | 首次需要下载 Embedding 模型 | 首次启动约 2-5 分钟，后续会命中缓存 |
| GPU 不可用 | nvidia-container-toolkit 未安装 | 注释掉 YAML 中 GPU 配置，使用 CPU 运行 |
| Docker daemon 未运行 | Docker Desktop 未启动 | 启动 Docker Desktop，或在终端执行 `start "" "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"` |

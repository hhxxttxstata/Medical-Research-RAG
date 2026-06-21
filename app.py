"""
FastAPI 应用入口
将 RAG 系统封装为可部署的 RESTful API 服务

接口列表:
  GET  /health              — 健康检查
  POST /documents/upload    — 上传文档并入库
  POST /chat                — 用户提问，返回 RAG/Agent 回答
  POST /diagnosis/predict   — 肺栓塞影像诊断（上传 NIfTI 进行预测）
  GET  /diagnosis/model     — 查看诊断模型状态
  GET  /logs                — 查看最近请求记录
"""

import asyncio
import concurrent.futures
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

# ── 安全防护 ────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent import Agent
from src.auth import verify_admin_api_key, verify_api_key

# ── 缓存 ────────────────────────────────────────────
from src.cache import CacheManager, RedisClient
from src.diagnosis import CTPADiagnosisModel, create_diagnosis_model
from src.document_loader import load_document

# ── 知识库管理 ──────────────────────────────────────
from src.knowledge_base import KnowledgeBase
from src.logger import get_logger
from src.memory import MemoryManager

# ── 可观测性 ────────────────────────────────────────
from src.monitoring.logging_config import setup_structlog
from src.monitoring.metrics import (
    get_metrics_registry,
)
from src.monitoring.tracing import init_tracing
from src.prompt_injection import detect_injection
from src.rag_pipeline import RAGPipeline

# ── Reranker ────────────────────────────────────────
from src.reranker import CrossEncoderReranker
from src.text_splitter import split_document
from src.watcher import DocumentWatcher

# ── 配置 ──────────────────────────────────────────────


class Settings:
    """应用配置"""

    data_dir: str = os.path.abspath("data")
    persist_dir: str = os.path.abspath("chroma_db")
    upload_dir: str = os.path.abspath("data")  # 上传文件存到 data/
    log_dir: str = os.path.abspath("logs")
    embedding_provider: str = "local"  # local | openai
    embedding_model: str | None = None  # 默认 BAAI/bge-small-zh-v1.5
    top_k: int = 5
    chunk_min_chars: int = 300
    chunk_max_chars: int = 500
    agent_mode: bool = True  # /chat 默认启用 Agent 模式


settings = Settings()


# ── 全局单例 ──────────────────────────────────────────

pipeline: RAGPipeline | None = None
agent: Agent | None = None
memory_manager: MemoryManager | None = None
diagnosis_model: CTPADiagnosisModel | None = None
cache_manager = None  # CacheManager 实例
reranker = None  # CrossEncoderReranker 实例
watcher = None  # DocumentWatcher 实例

# 诊断上传临时目录
_DIAGNOSIS_UPLOAD_DIR = os.path.abspath("data/diagnosis_uploads")


# ── 请求/响应模型 ────────────────────────────────────


class ChatRequest(BaseModel):
    model_config = {"strict": True}
    question: str = Field(..., min_length=1, description="用户提问")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索片段数")
    mode: str | None = Field(default="auto", description="模式: auto | rag | agent")
    report_type: str | None = Field(
        default=None,
        pattern=r"^(deployment|troubleshoot|meeting)$",
        description="强制报告类型（agent 模式时生效）",
    )


class ChatResponse(BaseModel):
    model_config = {"strict": True}
    success: bool
    answer: str
    mode: str
    sources: list = []
    elapsed: float = 0.0
    is_refusal: bool = False
    agent_info: dict | None = None
    process_log: list = []


class HealthResponse(BaseModel):
    model_config = {"strict": True}
    status: str
    version: str = "1.0.0"
    knowledge_base: dict | None = None
    timestamp: str = ""


class LogEntry(BaseModel):
    model_config = {"strict": True}
    timestamp: str
    question: str
    answer: str
    elapsed_seconds: float
    is_refusal: bool
    num_retrieved: int


class StatsResponse(BaseModel):
    model_config = {"strict": True}
    date: str
    total_queries: int
    success_count: int
    error_count: int
    refusal_count: int
    refusal_rate: float
    avg_response_time: float


# ── FastAPI 应用 ──────────────────────────────────────

app = FastAPI(
    title="RAG 知识库问答系统",
    description="基于检索增强生成的企业知识库 API，支持文档管理、智能问答、报告生成",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS（从环境变量读取允许的 origin，逗号分隔；留空或 * 时允许所有来源）
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
_cors_origins = (
    ["*"]
    if _cors_origins_env == "*" or not _cors_origins_env.strip()
    else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 速率限制 ──────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── MCP Server 挂载（可选） ───────────────────────────

_MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "false").lower() in ("true", "1")
_MCP_MOUNT_PATH: str = os.getenv("MCP_MOUNT_PATH", "/mcp")


# ── 生命周期 ──────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """应用启动时初始化 RAG 管道 + 缓存 + Reranker + 监听器"""
    global pipeline, agent, memory_manager, cache_manager, reranker, watcher

    # 初始化可观测性
    init_tracing(service_name="pe-rag-system-api")
    setup_structlog(service_name="pe-rag-system-api", force_json=False)

    print("\n" + "=" * 60)
    print("  🚀 RAG API 服务启动中...")
    print("=" * 60)

    # 确保目录存在
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.persist_dir, exist_ok=True)
    os.makedirs(settings.log_dir, exist_ok=True)

    # ── 初始化 Cross-encoder Reranker ──
    print("\n📊 初始化 Cross-encoder Reranker...")
    reranker = CrossEncoderReranker()
    reranker._load_model()

    # ── 初始化缓存 ──
    print("\n💾 初始化缓存系统...")
    RedisClient.get_client()  # 尝试连接 Redis（失败则降级内存）

    def _emb_fn(texts):
        if pipeline and pipeline.embedding_provider:
            return pipeline.embedding_provider.embed(texts)
        return [[0.0] * 768]

    cache_manager = CacheManager(embedding_fn=_emb_fn)
    # 将 embedding cache 注入 embedding_provider
    if pipeline:
        pipeline.embedding_provider._cache = cache_manager.embedding
    print("  ✅ 缓存就绪（Redis 优先 / 内存 fallback）")

    # 初始化管道（传入 reranker 和 cache）
    pipeline = RAGPipeline(
        data_dir=settings.data_dir,
        persist_dir=settings.persist_dir,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        chunk_min_chars=settings.chunk_min_chars,
        chunk_max_chars=settings.chunk_max_chars,
        enable_reranker=True,
        reranker=reranker if reranker.model_ready else None,
        cache_manager=cache_manager,
    )

    # 如果知识库为空，自动初始化
    count = pipeline.vector_store.count()
    if count == 0:
        print("\n📚 知识库为空，自动初始化...")
        pipeline.initialize_knowledge_base()
        count = pipeline.vector_store.count()
    else:
        print(f"\n📚 知识库已就绪: {count} 个 Chunk")

    # 初始化 Agent
    agent = Agent(rag_pipeline=pipeline)
    agent.tools["generate_report"].set_generator(pipeline.generator)

    # 初始化记忆系统（复用 embedding 模型做长期记忆向量化）
    print("\n🧠 初始化记忆系统...")
    memory_manager = MemoryManager(
        embedding_provider=pipeline.embedding_provider,
        persist_dir=settings.persist_dir,
    )
    agent.memory_manager = memory_manager
    print("  ✅ 三层记忆就绪（短期对话 / 工作进度 / 长期偏好）")

    # 预热 embedding 模型，避免第一请求卡顿
    print("\n🔋 预热 Embedding 模型...")
    try:
        pipeline.embedding_provider.warmup()
        print("  ✅ 模型预热完成")
    except AttributeError:
        # 旧版没有 warmup 方法
        pass
    except Exception as e:
        print(f"  ⚠️ 预热失败: {e}")

    # 初始化肺栓塞诊断模型
    print("\n🩺 初始化肺栓塞诊断模型...")
    _init_diagnosis_model()

    print(f"🔧 Embedding: {pipeline.embedding_provider.__class__.__name__}")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print(f"🤖 Agent 模式: {'启用' if settings.agent_mode else '关闭'}")
    print("📊 Metrics: /metrics")
    if _MCP_ENABLED:
        _mount_mcp_server(pipeline)

    # ── 启动文档监听器（增量索引） ──
    from src.watcher import ProcessedFilesTracker

    print("\n👀 启动文档监听器...")
    try:
        persist_path = os.path.join(settings.log_dir, ".processed_files.json")
        tracker = ProcessedFilesTracker(persist_path=persist_path)
        watcher = DocumentWatcher(pipeline, watch_dir=settings.data_dir, tracker=tracker)
        watcher.start()
        print("  ✅ 监听 data/ 目录，新增 PDF/MD/TXT 自动入库")
    except Exception as e:
        print(f"  ⚠️ 文档监听器启动失败: {e}")
        watcher = None

    print(f"🔧 Embedding: {pipeline.embedding_provider.__class__.__name__}")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print(f"🤖 Agent 模式: {'启用' if settings.agent_mode else '关闭'}")
    print(f"📊 Cross-encoder Reranker: {'启用' if reranker and reranker.model_ready else '未加载'}")
    print(f"💾 缓存: {'Redis' if RedisClient.is_enabled() else '内存 LRU'}")
    print("📊 Metrics: /metrics")
    if _MCP_ENABLED:
        _mount_mcp_server(pipeline)
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown():
    """清理资源"""
    global watcher
    if watcher:
        watcher.stop()
    print("👋 API 服务已关闭")


def _init_diagnosis_model():
    """初始化肺栓塞诊断模型（全局单例）"""
    global diagnosis_model
    diagnosis_model = create_diagnosis_model()
    if diagnosis_model and diagnosis_model.is_loaded:
        print("  ✅ 肺栓塞诊断模型已加载")
    else:
        err = diagnosis_model.load_error if diagnosis_model else "创建失败"
        print(f"  ⚠️  肺栓塞诊断模型未加载: {err}")
        print("  💡 设置 PE_MODEL_PATH 环境变量指定模型权重文件路径")


# ── 接口 1: 健康检查 ─────────────────────────────────


@limiter.limit("30/minute")
@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """健康检查 — 确认服务是否正常运行"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    kb_info = None
    try:
        count = pipeline.vector_store.count()
        kb_info = {
            "chunk_count": count,
            "data_dir": settings.data_dir,
            "embedding": pipeline.embedding_provider.__class__.__name__,
            "top_k": pipeline.top_k,
            "chunk_range": f"{pipeline.chunk_min_chars}-{pipeline.chunk_max_chars}",
        }
    except Exception:
        kb_info = {"error": "无法获取知识库状态"}

    return HealthResponse(
        status="ok",
        knowledge_base=kb_info,
        timestamp=datetime.now().isoformat(),
    )


# ── 接口 2: 上传文档 ─────────────────────────────────


@limiter.limit("5/minute")
@app.post("/documents/upload")
async def upload_document(
    request: Request,
    _: None = Depends(verify_api_key),
    file: UploadFile = File(..., description="要上传的文档（PDF/MD/TXT）"),
    auto_index: bool = Form(True, description="上传后自动入库"),
    rebuild: bool = Form(False, description="是否重建整个知识库索引"),
):
    """上传文档并（可选）将其纳入知识库

    支持格式: PDF, Markdown (.md), 纯文本 (.txt)
    文件会保存到 data/ 目录下，随后进行分块、向量化并存入 ChromaDB。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    # ── 1. 验证文件类型 ──
    filename = file.filename or f"upload_{int(time.time())}"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".md", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{suffix}'，支持 PDF/MD/TXT",
        )

    # ── 2. 保存文件 ──
    save_path = os.path.join(settings.upload_dir, filename)
    # 避免同名覆盖：加时间戳
    if os.path.exists(save_path):
        stem = Path(filename).stem
        save_path = os.path.join(
            settings.upload_dir,
            f"{stem}_{int(time.time())}{suffix}",
        )

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    file_size = len(content)

    # ── 3. 解析文档（快速验证可读性） ──
    try:
        doc = load_document(save_path)
    except Exception as e:
        # 解析失败，清理已保存的文件
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"文档解析失败: {str(e)}")

    result_info = {
        "filename": Path(save_path).name,
        "size_bytes": file_size,
        "file_type": suffix,
        "chars": len(doc["full_text"]),
        "pages": doc.get("total_pages", 1),
    }

    if not auto_index:
        return {
            "success": True,
            "message": "文件已保存，未入库（auto_index=False）",
            "file": result_info,
        }

    # ── 4. 入库（全量重建 或 增量添加） ──
    if rebuild:
        # 强制重建整个知识库
        pipeline.initialize_knowledge_base(force_reindex=True)
        count = pipeline.vector_store.count()
        message = "知识库已完整重建"
    else:
        # 增量添加：只处理新文件
        try:
            chunks = split_document(
                doc,
                chunk_min_chars=pipeline.chunk_min_chars,
                chunk_max_chars=pipeline.chunk_max_chars,
            )
            if not chunks:
                raise HTTPException(status_code=400, detail="文档切分结果为空")

            texts = [c["text"] for c in chunks]
            embeddings = pipeline.embedding_provider.embed(texts)
            pipeline.vector_store.add_chunks(chunks, embeddings)

            count = pipeline.vector_store.count()
            message = f"文档已入库，新增 {len(chunks)} 个 Chunk"
            result_info["new_chunks"] = len(chunks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")

    return {
        "success": True,
        "message": message,
        "file": result_info,
        "total_chunks": count,
    }


# ── 辅助：将同步阻塞调用放进线程池 ──────────────────

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)


async def run_blocking(fn, *args, **kwargs):
    """在默认线程池中执行同步阻塞函数，避免阻塞事件循环"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_THREAD_POOL, lambda: fn(*args, **kwargs))


# ── 接口 3: 聊天/问答 ────────────────────────────────


@limiter.limit("20/minute")
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _: None = Depends(verify_api_key)):
    """用户提问，返回 RAG 或 Agent 的回答

    三种模式:
    - auto  (默认): 自动判断意图，报告类走 Agent，其他走 RAG
    - rag:         强制走常规 RAG 问答
    - agent:       强制走 Agent 路由（含意图识别和工具调用）

    会话 ID 通过 X-Session-ID 请求头传入，无则自动生成。
    """
    if pipeline is None or agent is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    # ── 会话 ID 管理 ──
    session_id: str = request.headers.get(
        "X-Session-ID",
        request.headers.get("x-session-id", ""),
    )
    if not session_id:
        session_id = str(uuid.uuid4())

    start = time.time()
    question = req.question.strip()
    mode = (req.mode or "auto").lower()

    # ── 提示注入检测（医学 Q&A 安全防护） ──
    is_injection, injection_reason = detect_injection(question)
    if is_injection:
        logger = get_logger(log_dir=settings.log_dir)
        logger.log_query(
            question=question,
            retrieved_chunks=[],
            answer=f"[BLOCKED] {injection_reason}",
            elapsed=0.0,
            is_refusal=True,
        )
        return ChatResponse(
            success=False,
            answer=f"输入被拒绝：{injection_reason}",
            mode=mode,
            elapsed=0.0,
            is_refusal=True,
            process_log=[{"step": "安全检查", "detail": injection_reason, "status": "blocked"}],
        )

    if mode == "rag":
        # 强制 RAG 模式 — 全部在线程池执行
        log: list = []
        log.append({"step": "模式选择", "detail": "RAG 问答模式", "status": "ok"})
        log.append({"step": "检索知识库", "detail": f"top_k={req.top_k or settings.top_k}", "status": "running"})
        result = await run_blocking(pipeline.query, question, req.top_k)
        log.append(
            {"step": "检索知识库", "detail": f"检索到 {len(result.get('sources', []))} 个相关片段", "status": "ok"}
        )
        log.append(
            {
                "step": "生成回答",
                "detail": "LLM 混合模式生成完成（参考文档 + 自身知识）" if not result.get("is_refusal") else "触发拒答",
                "status": "ok",
            }
        )
        elapsed = round(time.time() - start, 2)
        log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})

        return ChatResponse(
            success=not result.get("error"),
            answer=result["answer"],
            mode="rag",
            sources=[
                {
                    "id": s["id"],
                    "filename": s["metadata"].get("filename", ""),
                    "page": s["metadata"].get("page", ""),
                    "score": round(s["score"], 3),
                    "text": s["text"][:200],
                }
                for s in result.get("sources", [])
            ],
            elapsed=elapsed,
            is_refusal=result.get("is_refusal", False),
            process_log=log,
        )

    elif mode == "agent":
        return await run_blocking(_sync_agent_mode, question, req.top_k, req.report_type, start, session_id)

    else:
        # "auto" 模式
        return await run_blocking(_sync_auto_mode, question, req.top_k, start, session_id)


def _sync_agent_mode(
    question: str, top_k: int, report_type: str | None, start: float, session_id: str = "default"
) -> ChatResponse:
    """同步：强制 Agent 模式"""
    log: list = []

    if report_type:
        log.append({"step": "模式选择", "detail": f"Agent 模式（指定报告类型: {report_type}）", "status": "ok"})
        log.append({"step": "意图识别", "detail": f"跳过识别，直接使用指定类型: {report_type}", "status": "ok"})
        k = top_k or 8
        log.append({"step": "检索知识库", "detail": f"top_k={k}", "status": "running"})
        retrieved = pipeline.retriever.retrieve(question, top_k=k)
        log.append({"step": "检索知识库", "detail": f"检索到 {len(retrieved)} 个相关片段", "status": "ok"})
        content_parts = []
        for i, chunk in enumerate(retrieved, 1):
            meta = chunk["metadata"]
            content_parts.append(f"### 来源 [{i}]: {meta.get('filename', '未知')}\n{chunk['text']}\n")
        content = "\n\n".join(content_parts)

        log.append({"step": "调用工具", "detail": f"generate_report (type={report_type})", "status": "running"})
        generator = agent.tools["generate_report"]
        generator.set_generator(pipeline.generator)
        tool_result = generator.run(
            report_type=report_type,
            content=content,
            topic=question,
            context={"sources_summary": "知识库检索", "source_count": len(retrieved)},
        )
        # 记录到长期记忆
        _remember_chat(session_id, question, tool_result.get("report", ""), "report_generate")
        log.append({"step": "调用工具", "detail": "报告生成完成", "status": "ok"})
        elapsed = round(time.time() - start, 2)
        log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
        return ChatResponse(
            success=tool_result.get("success", False),
            answer=tool_result.get("report", "报告生成失败"),
            mode="agent",
            elapsed=elapsed,
            agent_info={
                "intent": "report_generate",
                "report_type": report_type,
                "tool": "generate_report",
            },
            process_log=log,
        )

    # 自动意图识别（带 session_id 启用记忆）
    log.append({"step": "模式选择", "detail": "Agent 模式（自动识别意图）", "status": "ok"})
    log.append({"step": "意图识别", "detail": "分析中...", "status": "running"})
    intent_result = agent.process(question, session_id=session_id)
    elapsed = round(time.time() - start, 2)

    if intent_result.get("agent_handled"):
        intent_info = intent_result.get("intent", {})
        tool_name = intent_result.get("tool", "")
        tool_result = intent_result.get("result", {})

        # 提取意图字符串（兼容新旧格式）
        intent_str = intent_info if isinstance(intent_info, str) else intent_info.get("intent", "")

        # 诊断类工具：从 formatted_report 提取回答
        if tool_name == "diagnose_pulmonary_embolism":
            answer = tool_result.get("formatted_report", "诊断失败")
            log.append(
                {
                    "step": "意图识别",
                    "detail": f"识别为肺栓塞诊断（ReAct {intent_result.get('react_steps', 0)} 步）",
                    "status": "ok",
                }
            )
            log.append({"step": "调用工具", "detail": f"{tool_name} → 诊断完成", "status": "ok"})
            log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
            return ChatResponse(
                success=tool_result.get("success", False),
                answer=answer,
                mode="agent",
                elapsed=elapsed,
                agent_info={
                    "intent": "pe_diagnosis",
                    "tool": tool_name,
                    "diagnosis_result": {
                        "probability": tool_result.get("probability"),
                        "prediction": tool_result.get("prediction"),
                        "risk_level": tool_result.get("risk_level", "未知"),
                    },
                    "react_steps": intent_result.get("react_steps"),
                    "react_termination": intent_result.get("react_termination"),
                },
                process_log=log,
            )

        # 报告生成类工具
        answer = tool_result.get("report", "生成失败")
        log.append(
            {
                "step": "意图识别",
                "detail": f"识别为报告生成（{intent_result.get('report_type', '')}，ReAct {intent_result.get('react_steps', 0)} 步）",
                "status": "ok",
            }
        )
        log.append(
            {"step": "调用工具", "detail": f"generate_report → {intent_result.get('report_type', '')}", "status": "ok"}
        )
        log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
        return ChatResponse(
            success=tool_result.get("success", False),
            answer=answer,
            mode="agent",
            elapsed=elapsed,
            agent_info={
                "intent": intent_str,
                "report_type": intent_result.get("report_type"),
                "tool": intent_result.get("tool"),
                "confidence": (intent_info.get("confidence") if isinstance(intent_info, dict) else None),
                "react_steps": intent_result.get("react_steps"),
                "react_termination": intent_result.get("react_termination"),
            },
            process_log=log,
        )
    else:
        log.append({"step": "意图识别", "detail": "未匹配到专用工具", "status": "ok"})
        log.append({"step": "转为RAG问答", "detail": "fallback to RAG", "status": "running"})
        result = pipeline.query(question, top_k=top_k)
        log.append({"step": "转为RAG问答", "detail": "回答生成完成", "status": "ok"})
        log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
        return ChatResponse(
            success=not result.get("error"),
            answer=result["answer"],
            mode="rag",
            sources=[
                {
                    "id": s["id"],
                    "filename": s["metadata"].get("filename", ""),
                    "score": round(s["score"], 3),
                    "text": s["text"][:200],
                }
                for s in result.get("sources", [])
            ],
            elapsed=elapsed,
            is_refusal=result.get("is_refusal", False),
            agent_info={"intent": "normal_query", "fallback_to_rag": True},
            process_log=log,
        )


def _sync_auto_mode(question: str, top_k: int, start: float, session_id: str = "default") -> ChatResponse:
    """同步：auto 模式 — Agent 优先，Agent 不处理则回退 RAG"""
    log: list = []

    # Agent 自动判断（LLM 分类，规则兜底）
    log.append({"step": "模式选择", "detail": "Auto 模式（Agent 优先）", "status": "ok"})
    intent_result = agent.process(question, top_k=top_k or 8, session_id=session_id)

    if intent_result.get("agent_handled"):
        intent_info = intent_result.get("intent", {})
        tool_name = intent_result.get("tool", "")
        tool_result = intent_result.get("result", {})

        if tool_name == "diagnose_pulmonary_embolism":
            answer = tool_result.get("formatted_report", "诊断失败")
            log.append(
                {
                    "step": "意图识别",
                    "detail": f"识别为肺栓塞诊断（ReAct {intent_result.get('react_steps', 0)} 步）",
                    "status": "ok",
                }
            )
            log.append({"step": "调用工具", "detail": f"{tool_name} → 诊断完成", "status": "ok"})
            elapsed = round(time.time() - start, 2)
            log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
            return ChatResponse(
                success=tool_result.get("success", False),
                answer=answer,
                mode="agent",
                elapsed=elapsed,
                agent_info={
                    "intent": "pe_diagnosis",
                    "tool": tool_name,
                    "react_steps": intent_result.get("react_steps"),
                },
                process_log=log,
            )

        # 报告生成
        answer = tool_result.get("report", "生成失败")
        log.append(
            {
                "step": "意图识别",
                "detail": f"识别为报告生成（{intent_result.get('report_type', '')}，ReAct {intent_result.get('react_steps', 0)} 步）",
                "status": "ok",
            }
        )
        log.append(
            {"step": "调用工具", "detail": f"generate_report → {intent_result.get('report_type', '')}", "status": "ok"}
        )
        elapsed = round(time.time() - start, 2)
        log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
        return ChatResponse(
            success=tool_result.get("success", False),
            answer=answer,
            mode="agent",
            elapsed=elapsed,
            agent_info={
                "intent": str(intent_info.get("intent", "")),
                "report_type": intent_result.get("report_type"),
                "tool": tool_name,
            },
            process_log=log,
        )

    # Agent 不处理 → 回退到 RAG
    log.append({"step": "Agent判断", "detail": "未匹配专用工具，转为 RAG 问答", "status": "ok"})
    log.append({"step": "检索知识库", "detail": f"top_k={top_k or 5}", "status": "running"})
    result = pipeline.query(question, top_k=top_k)
    log.append({"step": "检索知识库", "detail": f"检索到 {len(result.get('sources', []))} 个相关片段", "status": "ok"})
    log.append(
        {
            "step": "生成回答",
            "detail": "LLM 混合模式生成完成" if not result.get("is_refusal") else "触发拒答",
            "status": "ok",
        }
    )
    elapsed = round(time.time() - start, 2)
    log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})
    return ChatResponse(
        success=not result.get("error"),
        answer=result["answer"],
        mode="rag",
        sources=[
            {
                "id": s["id"],
                "filename": s["metadata"].get("filename", ""),
                "score": round(s["score"], 3),
                "text": s["text"][:200],
            }
            for s in result.get("sources", [])
        ],
        elapsed=elapsed,
        is_refusal=result.get("is_refusal", False),
        process_log=log,
    )


# ── 接口 4: 肺栓塞诊断 ────────────────────────────────


class DiagnosisPredictResponse(BaseModel):
    model_config = {"strict": True}
    success: bool
    probability: float = 0.0
    prediction: int = 0
    threshold: float = 0.5
    risk_level: str = ""
    positive_voxel_ratio: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0
    filename: str = ""
    error: str | None = None
    visualization: dict | None = Field(default=None, description="base64 编码的可视化图像")


class DiagnosisModelStatus(BaseModel):
    model_config = {"strict": True}
    loaded: bool
    model_path: str | None = None
    device: str = ""
    input_shape: list = []
    threshold: float = 0.5
    error: str | None = None


@limiter.limit("5/minute")
@app.post("/diagnosis/predict", response_model=DiagnosisPredictResponse, response_model_exclude_unset=False)
async def diagnosis_predict(
    request: Request,
    _: None = Depends(verify_api_key),
    file: UploadFile = File(..., description="CTPA 影像文件（NIfTI 格式 .nii / .nii.gz）"),
):
    """上传 CTPA 影像（NIfTI 格式）进行肺栓塞诊断预测

    返回肺栓塞概率、风险等级和分割信息。
    需要先配置模型权重（PE_MODEL_PATH 环境变量）。
    """
    global diagnosis_model

    # 1. 检查模型是否加载
    if diagnosis_model is None or not diagnosis_model.is_loaded:
        _init_diagnosis_model()
        if diagnosis_model is None or not diagnosis_model.is_loaded:
            err = diagnosis_model.load_error if diagnosis_model else "模型初始化失败"
            return DiagnosisPredictResponse(
                success=False,
                error=f"诊断模型未加载: {err}。请设置 PE_MODEL_PATH 环境变量指向模型权重文件。",
            )

    # 2. 验证文件格式
    filename = file.filename or f"ctpa_{int(time.time())}.nii"
    suffix = Path(filename).suffix.lower()

    is_nii = suffix == ".nii" or ".nii" in [s.lower() for s in Path(filename).suffixes]
    if not is_nii and suffix != ".nii.gz":
        return DiagnosisPredictResponse(
            success=False,
            error=f"不支持的文件格式 '{suffix}'，仅支持 NIfTI 格式 (.nii / .nii.gz)",
        )

    # 3. 保存上传文件
    os.makedirs(_DIAGNOSIS_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(_DIAGNOSIS_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    # .nii.gz 的保存处理
    if filename.endswith(".gz") and not filename.endswith(".nii.gz"):
        save_path = os.path.join(_DIAGNOSIS_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}.nii.gz")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 4. 执行推理
    try:
        result = await run_blocking(diagnosis_model.predict, save_path, True)
    except Exception as e:
        # 清理临时文件
        try:
            os.remove(save_path)
        except Exception:
            pass
        return DiagnosisPredictResponse(success=False, error=f"推理失败: {str(e)}")

    # 5. 清理临时文件
    try:
        os.remove(save_path)
    except Exception:
        pass

    # 6. 构建响应
    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)

    if prob >= 0.9:
        risk = "高风险"
    elif prob >= 0.7:
        risk = "中风险"
    elif prob >= 0.5:
        risk = "低风险"
    else:
        risk = "阴性"

    return DiagnosisPredictResponse(
        success=result.get("success", False),
        probability=prob,
        prediction=pred,
        threshold=result.get("threshold", 0.5),
        risk_level=risk,
        positive_voxel_ratio=result.get("mask_positive_ratio", 0.0),
        inference_time=result.get("inference_time", 0.0),
        total_time=result.get("total_time", 0.0),
        filename=filename,
        error=result.get("error"),
        visualization=result.get("visualization"),  # 可选：阳性时返回可视化图像
    )


@limiter.limit("10/minute")
@app.get("/diagnosis/model", response_model=DiagnosisModelStatus)
async def diagnosis_model_status(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """查看肺栓塞诊断模型的状态"""
    if diagnosis_model:
        info = diagnosis_model.get_info()
        return DiagnosisModelStatus(
            loaded=info.get("loaded", False),
            model_path=info.get("model_path"),
            device=info.get("device", ""),
            input_shape=list(info.get("input_shape", [])),
            threshold=info.get("threshold", 0.5),
            error=info.get("load_error", None),
        )
    return DiagnosisModelStatus(
        loaded=False,
        error="诊断模型未初始化",
    )


# ── 接口 5: Prometheus 指标 ─────────────────────────


@limiter.limit("10/minute")
@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus 指标（供 Prometheus Server 抓取）"""
    return Response(content=generate_latest(get_metrics_registry()), media_type=CONTENT_TYPE_LATEST)


# ── 接口 6: 日志查询 ─────────────────────────────────


@limiter.limit("5/minute")
@app.get("/logs")
async def get_logs(
    request: Request,
    _: None = Depends(verify_admin_api_key),
    n: int = Query(10, ge=1, le=200, description="返回最近 N 条记录"),
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="指定日期 (YYYY-MM-DD)，默认今天"),
):
    """查看最近请求记录"""
    logger = get_logger(log_dir=settings.log_dir)

    if date:
        log_file = os.path.join(settings.log_dir, f"rag_{date}.jsonl")
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail=f"未找到 {date} 的日志文件")
        records = _read_jsonl(log_file, n)
    else:
        records = logger.get_recent_queries(n)

    # 脱敏和精简
    sanitized = []
    for r in records:
        sanitized.append(
            {
                "timestamp": r.get("timestamp", ""),
                "question": r.get("question", "")[:200],
                "elapsed_seconds": r.get("elapsed_seconds", 0),
                "is_refusal": r.get("is_refusal", False),
                "num_retrieved": r.get("num_retrieved", 0),
                "error": r.get("error"),
                "answer_preview": r.get("answer", "")[:300],
            }
        )

    # 统计摘要
    stats = logger.get_today_stats()

    return {
        "success": True,
        "total_queries_today": stats.get("total_queries", 0),
        "refusal_rate_today": f"{stats.get('refusal_rate', 0):.1f}%",
        "avg_response_time": f"{stats.get('avg_response_time', 0):.2f}s",
        "records": sanitized,
        "record_count": len(sanitized),
    }


# ── 附加接口: 系统统计 ──────────────────────────────


@limiter.limit("10/minute")
@app.get("/stats")
async def get_stats(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """查看系统运行统计"""
    logger = get_logger(log_dir=settings.log_dir)
    stats = logger.get_today_stats()

    # 计算检索质量汇总
    retrieval_quality = {}
    scores = stats.get("avg_retrieval_scores", [])
    if scores:
        retrieval_quality["avg_retrieval_score"] = round(sum(scores) / len(scores), 4)
        retrieval_quality["total_samples"] = len(scores)
    rates = stats.get("avg_overlap_rates", [])
    if rates:
        retrieval_quality["avg_overlap_rate"] = round(sum(rates) / len(rates), 4)

    return StatsResponse(
        date=stats.get("date", ""),
        total_queries=stats.get("total_queries", 0),
        success_count=stats.get("success_count", 0),
        error_count=stats.get("error_count", 0),
        refusal_count=stats.get("refusal_count", 0),
        refusal_rate=stats.get("refusal_rate", 0),
        avg_response_time=stats.get("avg_response_time", 0),
    )


# ── 辅助函数 ─────────────────────────────────────────


def _remember_chat(session_id: str, query: str, answer: str, intent: str = "normal_query") -> None:
    """将对话记录到记忆系统（供非 Agent 路径使用，如指定 report_type 的 Agent 模式）"""
    if memory_manager is None:
        return
    try:
        memory_manager.remember(
            session_id=session_id,
            query=query,
            answer=answer[:500],
            intent_info={"intent": intent},
        )
    except Exception:
        pass


def _read_jsonl(filepath: str, n: int) -> list:
    """读取 JSONL 文件，返回最近 N 条"""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-n:]


# ── 接口 7: 知识库管理 ────────────────────────────


@app.get("/knowledge-base/collections")
async def kb_list_collections(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """列出所有集合"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    cols = KnowledgeBase.list_collections(pipeline.vector_store)
    return {"success": True, "collections": cols}


@app.post("/knowledge-base/collections")
async def kb_create_collection(
    request: Request,
    _: None = Depends(verify_api_key),
    name: str = Form(..., description="集合名称"),
    tags: str = Form("", description="逗号分隔的标签"),
):
    """创建新集合"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = KnowledgeBase.create_collection(pipeline.vector_store, name, tags=tag_list)
    return {"success": True, "collection": result}


@app.delete("/knowledge-base/collections/{name}")
async def kb_delete_collection(
    name: str,
    request: Request,
    _: None = Depends(verify_api_key),
):
    """删除集合"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    ok = KnowledgeBase.delete_collection(pipeline.vector_store, name)
    return {"success": ok}


@app.get("/knowledge-base/collections/{name}")
async def kb_get_collection(
    name: str,
    request: Request,
    _: None = Depends(verify_api_key),
):
    """获取集合详情"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    info = KnowledgeBase.get_collection_info(pipeline.vector_store, name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"集合 '{name}' 不存在")
    return {"success": True, "collection": info}


@app.post("/knowledge-base/collections/{name}/rebuild")
async def kb_rebuild_collection(
    name: str,
    request: Request,
    _: None = Depends(verify_api_key),
):
    """重建集合索引"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    # 切换到目标集合
    old_name = pipeline.vector_store.collection_name
    pipeline.vector_store.collection_name = name
    try:
        count = pipeline.initialize_knowledge_base(force_reindex=True)
        version = KnowledgeBase.bump_version(pipeline.vector_store, name)
        return {"success": True, "total_chunks": count, "version": version}
    finally:
        pipeline.vector_store.collection_name = old_name


@app.get("/knowledge-base/tags")
async def kb_list_tags(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """列出所有标签"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    tags = KnowledgeBase.get_all_tags(pipeline.vector_store)
    return {"success": True, "tags": tags}


# ── 接口 8: 缓存管理 ──────────────────────────────


@app.post("/cache/clear")
async def cache_clear(
    request: Request,
    _: None = Depends(verify_admin_api_key),
):
    """清除所有缓存"""
    if cache_manager:
        cache_manager.invalidate_all()
    return {"success": True, "message": "缓存已清除"}


@app.get("/cache/status")
async def cache_status(
    request: Request,
    _: None = Depends(verify_api_key),
):
    """缓存状态"""
    from src.cache import RedisClient

    return {
        "success": True,
        "redis_connected": RedisClient.is_enabled(),
        "cache_type": "redis" if RedisClient.is_enabled() else "memory_lru",
    }


# ── MCP SSE 挂载 ────────────────────────────────────


def _mount_mcp_server(pipeline_instance) -> None:
    """将 MCP SSE Server 挂载到 FastAPI 应用的指定路径

    通过环境变量控制：
        MCP_ENABLED=true      开启 MCP SSE 挂载
        MCP_MOUNT_PATH=/mcp   挂载路径（默认 /mcp）

    MCP Host 连接时使用：
        SSE 端点: http://host:port/mcp/sse
        消息端点: http://host:port/mcp/messages/
    """
    try:
        from src.mcp_server import create_mcp_server

        mcp = create_mcp_server(server_name="pe-rag-system")
        sse_app = mcp.sse_app(mount_path=_MCP_MOUNT_PATH)
        app.mount(_MCP_MOUNT_PATH, sse_app)
        print(f"  🌐 MCP SSE: http://0.0.0.0:{int(os.getenv('API_PORT', '8000'))}{_MCP_MOUNT_PATH}/sse")
        print(f"  📪 MCP 消息: http://0.0.0.0:{int(os.getenv('API_PORT', '8000'))}{_MCP_MOUNT_PATH}/messages/")
    except Exception as e:
        print(f"  ⚠️  MCP Server 挂载失败: {e}")


# ── 直接运行 ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    print(f"\n🌐 启动 API 服务: http://{host}:{port}")
    print(f"📖 API 文档: http://{host}:{port}/docs")
    print()

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=["src", "data"] if reload else None,
    )

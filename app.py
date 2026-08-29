"""
FastAPI 应用入口 — 肺栓塞科研文献 RAG 问答助手
接口:
  GET  /health              — 健康检查
  POST /documents/upload    — 上传文档入库
  POST /chat                — RAG 问答（阻塞）
  POST /chat/stream         — RAG 问答（SSE 流式）
  POST /query               — Agentic RAG 查询（LangGraph runtime）
  GET  /logs                — 请求日志
  GET  /stats               — 运行统计
"""

import asyncio
import concurrent.futures
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _HAS_OTEL = True
except ImportError:
    trace = None  # type: ignore[assignment]
    FastAPIInstrumentor = None
    _HAS_OTEL = False

from src.auth import verify_admin_api_key, verify_api_key
from src.cache import CacheManager, RedisClient
from src.document_loader import load_document
from src.knowledge_base import KnowledgeBase
from src.logger import get_logger
from src.prompt_injection import detect_injection
from src.rag_pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.text_splitter import split_document
from src.watcher import DocumentWatcher

# ── 配置 ──────────────────────────────────────────────


class Settings:
    data_dir: str = os.path.abspath("data")
    upload_dir: str = os.path.abspath("data")
    log_dir: str = os.path.abspath("logs")
    embedding_provider: str = "local"
    embedding_model: str | None = None
    top_k: int = 8  # 检索深度：5 太小，同一文档的 chunk 会挤占其他来源
    chunk_min_chars: int = 300
    chunk_max_chars: int = 500
    vector_backend: str = "milvus"
    milvus_host: str = os.getenv("MILVUS_HOST", "milvus")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    # 本地免 Docker 模式：MILVUS_LITE=true python app.py（复用 milvus_db/ 本地文件库）
    milvus_lite: bool = os.getenv("MILVUS_LITE", "false").lower() == "true"
    # 服务配置对齐评测（evaluate.py：rewrite/rerank 均关闭才得到 80% Hit Rate）
    # rewrite：Step 1-7 消融证明无正收益，服务端冻结（省 1 次 LLM + 多路检索）
    # reranker：bge-reranker-v2-m3 CPU 上 10 对候选 ≈ 15s，默认关闭；
    #           GPU 部署时设 RERANKER_ENABLED=true 开启
    enable_rewrite: bool = False
    enable_reranker: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"


settings = Settings()

# ── 全局单例 ──────────────────────────────────────────

pipeline: RAGPipeline | None = None
cache_manager = None
reranker = None
watcher = None

# ── 请求/响应模型 ────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: str | None = Field(
        default=None,
        description="问答模式：rag（检索增强，默认）/ agent（LangGraph Agentic RAG）；auto 等价 rag",
    )
    domain: str | None = Field(
        default=None,
        description="知识域过滤：pe_literature（PE 文献）/ writing_guidelines（论文写作规范），None 表示全域",
    )
    session_id: str | None = Field(default=None, description="前端会话标识，响应中原样回传")


class ChatResponse(BaseModel):
    success: bool
    answer: str
    sources: list = []
    elapsed: float = 0.0
    is_refusal: bool = False
    process_log: list = []
    mode: str = "rag"
    agent_info: dict | None = None
    session_id: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    knowledge_base: dict | None = None
    timestamp: str = ""


class StatsResponse(BaseModel):
    date: str
    total_queries: int
    success_count: int
    error_count: int
    refusal_count: int
    refusal_rate: float
    avg_response_time: float


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: int = Field(..., ge=0, le=1, description="0=差 1=好")
    reason: str = Field(default="", max_length=100)
    message_id: str = Field(default="")


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Agentic RAG 查询问题")


# ── FastAPI 应用 ──────────────────────────────────────

app = FastAPI(
    title="肺栓塞科研文献 RAG 问答助手",
    description="面向肺栓塞中英文文献与论文写作规范的知识问答系统（仅科研辅助，不提供诊断建议）",
    version="2.0.0",
)

_cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenTelemetry 自动埋点（可选依赖） ────────────
_FastAPIInstrumentor = FastAPIInstrumentor  # name for is check
if _HAS_OTEL:
    _FastAPIInstrumentor.instrument_app(app)
    tracer = trace.get_tracer(__name__)
else:
    tracer = None  # type: ignore[assignment]

# ── 生命周期 ──────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global pipeline, cache_manager, reranker, watcher

    print("\n" + "=" * 60)
    print("  🚀 RAG API 服务启动中...")
    print("=" * 60)

    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.log_dir, exist_ok=True)

    # Reranker
    print("\n📊 初始化 Cross-encoder Reranker...")
    reranker = CrossEncoderReranker()
    reranker._load_model()

    # 缓存
    print("\n💾 初始化缓存系统...")
    RedisClient.get_client()

    def _emb_fn(texts):
        if pipeline and pipeline.embedding_provider:
            return pipeline.embedding_provider.embed(texts)
        return [[0.0] * 768]

    cache_manager = CacheManager(embedding_fn=_emb_fn)

    # 管道
    pipeline = RAGPipeline(
        data_dir=settings.data_dir,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        chunk_min_chars=settings.chunk_min_chars,
        chunk_max_chars=settings.chunk_max_chars,
        enable_rewrite=settings.enable_rewrite,
        enable_reranker=settings.enable_reranker,
        reranker=reranker if (reranker.model_ready and settings.enable_reranker) else None,
        cache_manager=cache_manager,
        vector_backend=settings.vector_backend,
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port,
        milvus_lite=settings.milvus_lite,
    )

    # 后台初始化知识库（不阻塞服务启动）

    def _async_init_kb():
        count = pipeline.vector_store.count()
        if count == 0:
            print("\n📚 知识库为空，后台初始化中...")
            pipeline.initialize_knowledge_base()
            cnt = pipeline.vector_store.count()
            print(f"\n📚 知识库初始化完成: {cnt} 个 Chunk")
        else:
            print(f"\n📚 知识库已就绪: {count} 个 Chunk")

    threading.Thread(target=_async_init_kb, daemon=True).start()

    # 将 cache 注入 embedding provider
    if pipeline:
        pipeline.embedding_provider._cache = cache_manager.embedding
    print("  ✅ 缓存就绪")

    # 预热
    print("\n🔋 预热 Embedding 模型...")
    try:
        pipeline.embedding_provider.warmup()
        print("  ✅ 模型预热完成")
    except AttributeError:
        pass
    except Exception as e:
        print(f"  ⚠️ 预热失败: {e}")

    # 文档监听器
    from src.watcher import ProcessedFilesTracker

    print("\n👀 启动文档监听器...")
    try:
        tracker = ProcessedFilesTracker(persist_path=os.path.join(settings.log_dir, ".processed_files.json"))
        watcher = DocumentWatcher(pipeline, watch_dir=settings.data_dir, tracker=tracker)
        watcher.start()
        print("  ✅ 监听 data/ 目录")
    except Exception as e:
        print(f"  ⚠️ 文档监听器启动失败: {e}")

    print(f"\n🔧 Embedding: {pipeline.embedding_provider.__class__.__name__}")
    print(f"🎯 Top-K: {pipeline.top_k}")
    print(f"📊 Reranker: {'启用' if reranker and reranker.model_ready else '未加载'}")
    print(f"💾 缓存: {'Redis' if RedisClient.is_enabled() else '内存 LRU'}")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown():
    global watcher
    if watcher:
        watcher.stop()
    print("👋 API 服务已关闭")


# ── 接口 1: 健康检查 ─────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    try:
        count = pipeline.vector_store.count()
        kb_info = {
            "chunk_count": count,
            "embedding": pipeline.embedding_provider.__class__.__name__,
            "top_k": pipeline.top_k,
        }
    except Exception:
        kb_info = None

    return HealthResponse(status="ok", knowledge_base=kb_info, timestamp=datetime.now().isoformat())


# ── 接口 2: 上传文档 ─────────────────────────────────


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    auto_index: bool = Form(True),
    rebuild: bool = Form(False),
    _: None = Depends(verify_api_key),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    # basename 净化，防路径穿越：反斜杠统一转正斜杠（跨平台，Windows/Linux 都有效）
    filename = Path((file.filename or f"upload_{int(time.time())}").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".md", ".txt"):
        raise HTTPException(status_code=400, detail=f"不支持 {suffix}，支持 PDF/MD/TXT")

    save_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(save_path):
        save_path = os.path.join(settings.upload_dir, f"{Path(filename).stem}_{int(time.time())}{suffix}")

    # 流式读取 + 大小限制，防内存 DoS（50 MB 上限）
    max_bytes = 50 * 1024 * 1024
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="文件过大，上限 50 MB")
    content = bytes(content)
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        doc = load_document(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"文档解析失败: {str(e)}")

    result_info = {
        "filename": Path(save_path).name,
        "size_bytes": len(content),
        "chars": len(doc["full_text"]),
    }

    if not auto_index:
        return {"success": True, "message": "文件已保存，未入库", "file": result_info}

    if rebuild:
        pipeline.initialize_knowledge_base(force_reindex=True)
        count = pipeline.vector_store.count()
        return {"success": True, "message": "知识库已重建", "total_chunks": count}
    else:
        chunks = split_document(doc, chunk_min_chars=pipeline.chunk_min_chars, chunk_max_chars=pipeline.chunk_max_chars)
        texts = [c["text"] for c in chunks]
        embeddings = pipeline.embedding_provider.embed(texts)
        pipeline.vector_store.add_chunks(chunks, embeddings)
        return {
            "success": True,
            "message": f"新增 {len(chunks)} 个 Chunk",
            "file": {**result_info, "new_chunks": len(chunks)},
            "total_chunks": pipeline.vector_store.count(),
        }


# ── 辅助线程池 ──────────────────────────────────────

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)


async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_THREAD_POOL, lambda: fn(*args, **kwargs))


# ── 接口 3: RAG 问答（阻塞） ─────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, _: None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    start = time.time()
    question = req.question.strip()

    # 提示注入检测
    is_injection, injection_reason = detect_injection(question)
    if is_injection:
        return ChatResponse(
            success=False,
            answer=f"输入被拒绝：{injection_reason}",
            elapsed=0.0,
            is_refusal=True,
            process_log=[{"step": "安全检查", "detail": injection_reason, "status": "blocked"}],
        )

    # Agentic 模式：路由到 LangGraph Agentic RAG（与 POST /query 同一服务）
    if req.mode == "agent":
        try:
            result = await run_blocking(_get_agent_service().query, question)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agentic 查询失败: {e}")
        elapsed = round(time.time() - start, 2)
        abstain = result.get("status") == "ABSTAIN"
        route = result.get("route", [])
        return ChatResponse(
            success=bool(result.get("answer")),
            answer=result.get("answer", ""),
            sources=result.get("evidence", []),
            elapsed=elapsed,
            is_refusal=abstain,
            mode="agent",
            agent_info={
                "status": result.get("status"),
                "route": route,
                "iterations": result.get("iterations", 0),
                "grader_called": result.get("grader_called", False),
                "latency_ms": result.get("latency_ms", 0),
                "abstain_reason": result.get("abstain_reason"),
            },
            session_id=req.session_id or "",
            process_log=[
                {"step": "Agentic 决策", "detail": " → ".join(route) or "-", "status": "ok"},
                {"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"},
            ],
        )

    log: list = []
    log.append(
        {
            "step": "检索知识库",
            "detail": f"top_k={req.top_k or settings.top_k}{' / domain=' + req.domain if req.domain else ''}",
            "status": "running",
        }
    )
    result = await run_blocking(pipeline.query, question, req.top_k, req.domain)
    elapsed = round(time.time() - start, 2)

    log.append(
        {
            "step": "检索知识库",
            "detail": f"检索到 {len(result.get('sources', []))} 个相关片段",
            "status": "ok",
        }
    )
    log.append(
        {
            "step": "生成回答",
            "detail": "完成" if not result.get("is_refusal") else "触发拒答",
            "status": "ok",
        }
    )
    log.append({"step": "完成", "detail": f"总耗时 {elapsed}s", "status": "ok"})

    return ChatResponse(
        success=not result.get("error"),
        answer=result["answer"],
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
        session_id=req.session_id or "",
    )


# ── 接口 4: RAG 流式问答（SSE） ────────────────────────


def _sse(event: str, data) -> str:
    return json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, _: None = Depends(verify_api_key)):
    """流式 SSE 端点

    后端在后台线程执行 query_stream（分阶段：检索→生成→token），
    所有事件通过 SSE 推送给前端。req.mode == "agent" 时路由到
    LangGraph Agentic RAG（一次性 answer，无 token 流）。

    事件类型:
      - status:  过程状态信息（检索中/已找到 N 条/生成中）
      - token:   LLM 的一个 token
      - answer:  一次性推完整回答（拒答或缓存命中时）
      - sources: 检索到的来源
      - agent_info: agent 模式的决策元信息（status/route/iterations 等）
      - elapsed: 总耗时
      - error:   错误消息
      - done:    结束标记
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")

    question = req.question.strip()

    is_injection, injection_reason = detect_injection(question)
    if is_injection:

        async def reject():
            yield f"data: {_sse('error', f'输入被拒绝: {injection_reason}')}\n\n"
            yield f"data: {_sse('done', '')}\n\n"

        return StreamingResponse(reject(), media_type="text/event-stream")

    # Agentic 模式：路由到 LangGraph Agentic RAG（一次性返回，无 token 流）
    if req.mode == "agent":

        async def agent_generate():
            loop = asyncio.get_running_loop()
            yield f"data: {_sse('status', 'Agentic 决策中（LangGraph runtime）')}\n\n"
            try:
                result = await loop.run_in_executor(_THREAD_POOL, lambda: _get_agent_service().query(question))
            except Exception as e:  # noqa: BLE001 — 错误经 SSE 下发而非中断流
                yield f"data: {_sse('error', f'Agentic 查询失败: {e}')}\n\n"
                yield f"data: {_sse('done', '')}\n\n"
                return
            agent_info = {
                k: result.get(k)
                for k in ("status", "route", "iterations", "grader_called", "latency_ms", "abstain_reason")
            }
            yield f"data: {_sse('agent_info', agent_info)}\n\n"
            yield f"data: {_sse('sources', result.get('evidence', []))}\n\n"
            yield f"data: {_sse('answer', result.get('answer', ''))}\n\n"
            yield f"data: {_sse('elapsed', round(result.get('latency_ms', 0) / 1000, 2))}\n\n"
            yield f"data: {_sse('done', '')}\n\n"

        return StreamingResponse(agent_generate(), media_type="text/event-stream")

    async def generate():
        loop = asyncio.get_running_loop()
        gen = pipeline.query_stream(question, req.top_k, req.domain)
        while True:
            ev = await loop.run_in_executor(_THREAD_POOL, lambda: next(gen, None))
            if ev is None:
                break
            yield f"data: {_sse(ev['event'], ev['data'])}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 接口 5: 日志查询 ─────────────────────────────────


@app.get("/logs")
async def get_logs(
    _: None = Depends(verify_admin_api_key),
    n: int = Query(10, ge=1, le=200),
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    logger = get_logger(log_dir=settings.log_dir)

    if date:
        log_file = os.path.join(settings.log_dir, f"rag_{date}.jsonl")
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail=f"未找到 {date} 的日志")
        records = _read_jsonl(log_file, n)
    else:
        records = logger.get_recent_queries(n)

    sanitized = [
        {
            "timestamp": r.get("timestamp", ""),
            "question": r.get("question", "")[:200],
            "elapsed_seconds": r.get("elapsed_seconds", 0),
            "is_refusal": r.get("is_refusal", False),
            "num_retrieved": r.get("num_retrieved", 0),
            "error": r.get("error"),
            "answer_preview": r.get("answer", "")[:300],
        }
        for r in records
    ]

    stats = logger.get_today_stats()
    return {
        "success": True,
        "total_queries_today": stats.get("total_queries", 0),
        "refusal_rate_today": f"{stats.get('refusal_rate', 0):.1f}%",
        "avg_response_time": f"{stats.get('avg_response_time', 0):.2f}s",
        "records": sanitized,
    }


@app.get("/stats")
async def get_stats(_: None = Depends(verify_api_key)):
    logger = get_logger(log_dir=settings.log_dir)
    stats = logger.get_today_stats()
    return StatsResponse(
        date=stats.get("date", ""),
        total_queries=stats.get("total_queries", 0),
        success_count=stats.get("success_count", 0),
        error_count=stats.get("error_count", 0),
        refusal_count=stats.get("refusal_count", 0),
        refusal_rate=stats.get("refusal_rate", 0),
        avg_response_time=stats.get("avg_response_time", 0),
    )


# ── 接口 7: 知识库管理 ──────────────────────────────


@app.get("/knowledge-base/collections")
async def kb_list_collections(_: None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    return {"success": True, "collections": KnowledgeBase.list_collections(pipeline.vector_store)}


@app.get("/knowledge-base/tags")
async def kb_list_tags(_: None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    return {"success": True, "tags": KnowledgeBase.get_all_tags(pipeline.vector_store)}


# ── 接口 8: Bad Case 反馈 ─────────────────────────


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """用户反馈（👍/👎），存入 logs/feedback.csv"""
    feedback_dir = os.path.join(settings.log_dir)
    os.makedirs(feedback_dir, exist_ok=True)
    path = os.path.join(feedback_dir, "feedback.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "question", "answer", "rating", "reason", "message_id"])
        writer.writerow(
            [
                datetime.now().isoformat(),
                req.question,
                req.answer,
                req.rating,
                req.reason,
                req.message_id,
            ]
        )
    return {"success": True}


@app.get("/feedback")
async def list_feedback(
    n: int = Query(50, ge=1, le=500),
    _: None = Depends(verify_admin_api_key),
):
    """获取用户反馈列表（管理用）"""
    path = os.path.join(settings.log_dir, "feedback.csv")
    if not os.path.exists(path):
        return {"success": True, "records": []}
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return {"success": True, "records": records[-n:]}


# ── Agentic Query 服务（Step 17：研究层 → 服务层） ────────────
# 薄适配：POST /query → LangGraph Agentic RAG（v2，Step 16 parity 已验证）
# 懒加载（首次调用 ~10-20s 加载 reranker/agent），复用服务的 Milvus 连接。


_agent_service = None


def _get_agent_service():
    """懒创建 AgentQueryService（构造本身不加载模型，query 时才加载）"""
    global _agent_service
    if _agent_service is None:
        from src.agent_service import AgentQueryService

        _agent_service = AgentQueryService(pipeline)
    return _agent_service


@app.post("/query")
async def agent_query(req: AgentQueryRequest, _: None = Depends(verify_api_key)):
    """Agentic RAG 查询（LangGraph runtime）

    返回结构化响应：answer / status(ACCEPT|ABSTAIN) / evidence / route /
    grader_called / latency_ms。首次调用含模型加载（10-20s）。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="服务尚未初始化完成")
    try:
        svc = _get_agent_service()
        result = await run_blocking(svc.query, req.question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic 查询失败: {e}")
    return result


# ── 辅助 ──────────────────────────────────────────────


def _read_jsonl(filepath: str, n: int) -> list:
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


# ── 直接运行 ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"\n🌐 启动 API: http://{host}:{port}")
    print(f"📖 API 文档: http://{host}:{port}/docs\n")

    uvicorn.run("app:app", host=host, port=port)

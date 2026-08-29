@echo off
rem ═══════════════════════════════════════════════════════════
rem  本地开发模式启动（无需 Docker）
rem  后端:  http://localhost:8000  (Swagger: /docs)
rem  前端:  http://localhost:3000
rem
rem  说明:
rem   - MILVUS_LITE=true 复用 milvus_db/ 本地向量库（无需起 Milvus 容器）
rem   - Redis 连不上时自动降级为内存缓存（无影响）
rem   - 依赖 .env 中的 DEEPSEEK_API_KEY 才能生成回答
rem   - 关闭窗口即停止对应服务
rem ═══════════════════════════════════════════════════════════

echo [1/2] 启动后端 (FastAPI :8000)...
start "PE-Backend :8000" cmd /k "set MILVUS_LITE=true&& set HF_HUB_OFFLINE=1&& .venv\Scripts\python.exe run.py"

echo [2/2] 启动前端 (Next.js :3000)...
pushd frontend
start "PE-Frontend :3000" cmd /k "pnpm dev"
popd

echo.
echo 启动中（首次加载 embedding/reranker 模型约 1-2 分钟，后端窗口出现 "Uvicorn running" 即可访问）...
echo   前端:  http://localhost:3000
echo   后端:  http://localhost:8000/docs
echo   健康:  http://localhost:8000/health
pause

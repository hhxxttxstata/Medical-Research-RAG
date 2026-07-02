@echo off
chcp 65001 >nul
title 肺栓塞智能问诊系统 — 一键启动

echo ════════════════════════════════════════════
echo  🩺 肺栓塞 RAG 智能问诊系统
echo ════════════════════════════════════════════
echo.

cd /d "%~dp0"

rem ── 检查 .env 是否有 LLM API Key ─────────────
findstr "DEEPSEEK_API_KEY=sk-" .env >nul 2>&1
if errorlevel 1 (
    findstr "SILICON_API_KEY=sk-" .env >nul 2>&1
    if errorlevel 1 (
        findstr "OPENAI_API_KEY=sk-" .env >nul 2>&1
        if errorlevel 1 (
            echo ⚠️  未检测到 LLM API Key！
            echo.
            echo   编辑 .env 文件，至少配置一种 LLM：
            echo   - DEEPSEEK_API_KEY （推荐，国内直连）
            echo   - SILICON_API_KEY
            echo   - OPENAI_API_KEY
            echo.
            pause
        )
    )
)

rem ── 启动后端 API（单独窗口） ────────────────
echo 📡 启动后端 API 服务...
start "RAG 后端" cmd /c "title RAG 后端 && python app.py & pause"

rem 等后端就绪（最多 60 秒）
echo ⏳ 等待后端就绪...
for /l %%i in (1,1,60) do (
    >nul 2>&1 curl -s http://localhost:8000/health || (
        timeout /t 1 /nobreak >nul
        if %%i equ 60 (
            echo ❌ 后端启动超时，请检查日志
            pause
            exit /b 1
        )
        if %%i equ 1 echo   .
        if %%i equ 20 echo   .. 仍在等待（首次启动需下载模型，约 2-3 分钟）
        if %%i equ 40 echo   ...
    )
)

echo ✅ 后端就绪！

rem ── 启动前端 Gradio（单独窗口） ────────────
echo 🌐 启动前端页面...
start "RAG 前端" cmd /c "title RAG 前端 && python gradio_app.py & pause"

rem 等前端就绪
timeout /t 3 /nobreak >nul

echo.
echo ════════════════════════════════════════════
echo  ✅ 系统启动完成！
echo.
echo  🌐 前端页面:   http://localhost:7860
echo  📡 后端 API:   http://localhost:8000
echo  📖 API 文档:   http://localhost:8000/docs
echo ════════════════════════════════════════════
echo.
echo 关闭后端或前端窗口即可停止服务。
echo 双击 deploy_docker.bat 可切换到 Docker 部署。
echo.
pause

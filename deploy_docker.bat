@echo off
chcp 65001 >nul
title 肺栓塞智能问诊系统 — Docker 部署

echo ════════════════════════════════════════════
echo  🐳 肺栓塞 RAG 系统 — Docker 部署
echo ════════════════════════════════════════════
echo.

cd /d "%~dp0"

where docker >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker 未安装
    echo   下载: https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe
    echo   安装后重试
    pause
    exit /b 1
)

echo 🔨 构建镜像并启动（首次 5-10 分钟）...
echo.

docker compose up -d --build

if errorlevel 0 (
    echo.
    echo ============================================
    echo  🌐 访问地址:
    echo   前端: http://localhost:7860
    echo   后端: http://localhost:8000
    echo ============================================
    echo.
    echo 查看日志: docker compose logs -f
    echo 停止服务: docker compose down
) else (
    echo ❌ 部署失败
    echo   如果网络不通，请直接双击 start.bat 启动
)
pause

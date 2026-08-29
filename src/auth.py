"""
src/auth.py — API Key 认证

设计决策：
  - 单一 Bearer token，从环境变量 API_KEY 读取
  - 装饰器式 Depends()，精确控制哪些接口需要认证
  - 不引入用户/角色体系，保持最小可行认证
"""

import os
from typing import Annotated

from fastapi import Header, HTTPException, status

# ── 配置 ────────────────────────────────────────────

_API_KEY: str | None = None


def load_api_key() -> str | None:
    """懒加载 API_KEY（仅在需要认证时读取 env）"""
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.getenv("API_KEY")
    return _API_KEY


# ── 依赖 ────────────────────────────────────────────


async def verify_api_key(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """FastAPI Depends: 验证 Bearer token

    用法:
        @app.post("/chat")
        async def chat(..., _: None = Depends(verify_api_key)):
            ...

    不需要认证的端点不注入此依赖。
    """
    api_key = load_api_key()
    if api_key is None:
        # 未配置 API_KEY 时，不做认证（兼容开发环境）
        return

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 只支持 Bearer scheme
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    if token != api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


async def verify_admin_api_key(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Admin 级认证（当前复用 API_KEY，保留独立函数以便扩展）"""
    await verify_api_key(authorization)

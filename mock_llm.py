"""兼容 shim —— Mock LLM 层的实现已迁移到 `app/mock_llm.py`。

保留本文件是为了让 `app.py`（HuggingFace Spaces / Render 的 Gradio Demo）
等既有入口继续用原有的 `import mock_llm` 路径，不必改动部署脚本。

新代码请直接使用其中一种方式：
- 设 `MOCK_LLM=1` 环境变量（推荐，与导入顺序无关）；
- 或 `from app.mock_llm import install; install()`。
"""
from __future__ import annotations

from app.mock_llm import (  # noqa: F401
    get_mock_chat_model,
    get_mock_structured_model,
    install,
)

__all__ = ["install", "get_mock_chat_model", "get_mock_structured_model"]

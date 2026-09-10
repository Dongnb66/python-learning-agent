"""LLM 客户端封装。

统一通过 LangChain 的 ChatOpenAI 接入任意 OpenAI 协议兼容端点
（DeepSeek / OpenAI / 本地 Ollama 等），并支持结构化输出。

注意：所有调用都走环境变量里的 llm_* 配置，不写死任何密钥。

Mock 模式
---------
设置环境变量 `MOCK_LLM=1` 即进入 Mock 模式（无需任何 Key、不访问网络），
由 `app.mock_llm` 返回结构合法的样例输出，便于离线跑通全流程与评测。

为什么在函数内部判断而不是替换模块符号：
各 Agent 用的是 `from app.llm import get_structured_model` 这种名字绑定，
导入后再打猴子补丁对它们无效；把判断放在函数体内则与导入顺序无关。
"""
import os
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings

_TRUTHY = {"1", "true", "yes", "on"}


def mock_enabled() -> bool:
    """是否启用 Mock 模式（`MOCK_LLM=1`）。"""
    return (os.getenv("MOCK_LLM") or "").strip().lower() in _TRUTHY


def get_chat_model(
    temperature: float | None = None, max_tokens: int | None = None
) -> ChatOpenAI:
    """返回一个 ChatOpenAI 实例。

    若不传 temperature / max_tokens，则使用配置文件中的默认值。
    """
    if mock_enabled():  # pragma: no cover - 由 mock 专属用例覆盖
        from app.mock_llm import get_mock_chat_model

        return get_mock_chat_model()  # type: ignore[return-value]

    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_tokens=s.llm_max_tokens if max_tokens is None else max_tokens,
        api_key=s.llm_api_key,
        base_url=s.llm_base_url,
        timeout=60,
        max_retries=2,
    )


def get_structured_model(
    schema: Any,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """返回绑定了结构化输出的聊天模型。

    强制使用 function calling：部分 OpenAI 兼容端点（如百炼 MaaS 的 deepseek-v3）
    在 json 模式下会把结果包进 ```json 代码块，导致解析失败；function calling
    让模型以工具调用参数的形式返回，LangChain 可稳定解析为 Pydantic 模型。
    """
    if mock_enabled():  # pragma: no cover - 由 mock 专属用例覆盖
        from app.mock_llm import get_mock_structured_model

        return get_mock_structured_model(schema)

    return get_chat_model(temperature=temperature, max_tokens=max_tokens).with_structured_output(
        schema, method="function_calling"
    )

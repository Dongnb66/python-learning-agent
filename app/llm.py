"""LLM 客户端封装。

统一通过 LangChain 的 ChatOpenAI 接入任意 OpenAI 协议兼容端点
（DeepSeek / OpenAI / 本地 Ollama 等），并支持结构化输出。

注意：所有调用都走环境变量里的 llm_* 配置，不写死任何密钥。
"""
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model(
    temperature: float | None = None, max_tokens: int | None = None
) -> ChatOpenAI:
    """返回一个 ChatOpenAI 实例。

    若不传 temperature / max_tokens，则使用配置文件中的默认值。
    """
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
    return get_chat_model(temperature=temperature, max_tokens=max_tokens).with_structured_output(
        schema, method="function_calling"
    )

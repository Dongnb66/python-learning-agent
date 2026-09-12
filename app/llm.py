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

from app import telemetry
from app.config import get_settings

_TRUTHY = {"1", "true", "yes", "on"}


def mock_enabled() -> bool:
    """是否启用 Mock 模式（`MOCK_LLM=1`）。"""
    return (os.getenv("MOCK_LLM") or "").strip().lower() in _TRUTHY


class InstrumentedModel:
    """给模型套一层**只负责计数与计时**的代理。

    为什么在工厂函数里包而不是改各 Agent 的调用点：
    调用点有 6 处（5 个单调用节点 + 自主循环），逐处埋点既散又容易漏；
    在唯一的出口 `get_chat_model` / `get_structured_model` 包一层，
    「模型被调用了几次、各花了多久、错了几次」就自动全覆盖。

    只拦截 `invoke`（本项目全部走同步 invoke）与 `bind_tools`
    （自主 Agent 的 ReAct 循环要用，必须保证返回值同样是代理，否则
    工具调用循环那一支会漏统计）。其余属性一律透传，LangChain 的
    Runnable 协议（with_structured_output / with_retry 等）不受影响。

    局限（如实说明）：`stream` / `batch` / `ainvoke` 未计数。当前项目
    全部使用同步 `invoke`，需要时再补。

    真实 / 模拟分开记：`llm_calls_total` 记全部调用，`llm_mock_calls_total`
    记其中属于 Mock 模型的部分 —— 真实调用数 = 两者之差。这样离线跑出来的
    指标依然可测（否则零 Key 环境下这个计数器永远是 0，等于没法验证），
    同时不会被误读成「真的调了这么多次 API」。
    """

    __slots__ = ("_inner", "_label", "_mock")

    def __init__(self, inner: Any, label: str, mock: bool = False) -> None:
        self._inner = inner
        self._label = label
        self._mock = mock

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        telemetry.bump(telemetry.C_LLM_CALLS)
        if self._mock:
            telemetry.bump(telemetry.C_LLM_MOCK_CALLS)
        try:
            with telemetry.span(f"llm.{self._label}", mock=self._mock):
                return self._inner.invoke(*args, **kwargs)
        except BaseException:
            telemetry.bump(telemetry.C_LLM_ERRORS)
            raise

    def bind_tools(self, *args: Any, **kwargs: Any) -> "InstrumentedModel":
        # 关键：返回包装后的对象，否则 ReAct 循环内每轮 invoke 都不计数
        return InstrumentedModel(self._inner.bind_tools(*args, **kwargs), self._label, self._mock)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def _instrument(inner: Any, label: str, *, mock: bool = False) -> InstrumentedModel:
    return InstrumentedModel(inner, label, mock)


def get_chat_model(
    temperature: float | None = None, max_tokens: int | None = None
) -> ChatOpenAI:
    """返回一个 ChatOpenAI 实例（已套可观测代理）。

    若不传 temperature / max_tokens，则使用配置文件中的默认值。
    """
    if mock_enabled():  # pragma: no cover - 由 mock 专属用例覆盖
        from app.mock_llm import get_mock_chat_model

        return get_mock_chat_model()  # type: ignore[return-value]

    s = get_settings()
    return _instrument(
        ChatOpenAI(
            model=s.llm_model,
            temperature=s.llm_temperature if temperature is None else temperature,
            max_tokens=s.llm_max_tokens if max_tokens is None else max_tokens,
            api_key=s.llm_api_key,
            base_url=s.llm_base_url,
            timeout=60,
            max_retries=2,
        ),
        "chat",
    )  # type: ignore[return-value]


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

        return _instrument(
            get_mock_structured_model(schema),
            getattr(schema, "__name__", "structured"),
            mock=True,
        )

    inner = get_chat_model(temperature=temperature, max_tokens=max_tokens).with_structured_output(
        schema, method="function_calling"
    )
    return _instrument(inner, getattr(schema, "__name__", "structured"))

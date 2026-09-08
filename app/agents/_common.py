"""智能体通用工具。"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.models import Message


def to_lc_messages(messages: list[Message]) -> list[BaseMessage]:
    """把内部 Message 列表转成 LangChain 消息对象。"""
    lc: list[BaseMessage] = []
    for m in messages:
        if m.role == "assistant":
            lc.append(AIMessage(content=m.content))
        else:
            lc.append(HumanMessage(content=m.content))
    return lc


def dialogue_text(messages: list[Message]) -> str:
    """把对话历史拼成纯文本，供正则提取 / 拼接提示使用。"""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)

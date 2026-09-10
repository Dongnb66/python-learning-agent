"""Agent 工具层 —— 供自主辅导智能体（ReAct）在运行时自行决定调用。

设计要点（面试可讲）：
1. **只读、无副作用**：全部是查询类工具，LLM 可以反复、任意顺序调用而不会破坏数据，
   这是允许「自主循环」的前提条件。
2. **零 LLM 依赖**：工具本身不调模型，可脱离 Agent 独立单测（与 mcp-toolkit 的
   lib 层「纯函数、可独立测试」同一思路）。
3. **返回结构化文本而非原始对象**：把 DB / 检索结果渲染成短文本，控制上下文长度，
   避免把整个 JSON 塞进 prompt 造成 token 浪费与干扰。

与 mcp-toolkit 的关系：那里是把能力标准化成 MCP 协议给外部宿主用；
这里是把能力标准化成 LangChain Tool 给本项目内部的 Agent 用——同一套分层思想。
"""
from __future__ import annotations

from langchain_core.tools import tool

from app import db
from app.rag import retrieve

_MAX_SNIPPET = 240
_MAX_JSON = 600


def _clip(text: str, n: int) -> str:
    """把任意文本压成单行并截断，控制进入上下文的信息量。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= n else flat[:n] + "…"


# --------------------------------------------------------------------------- #
# 工具 1：RAG 检索真实学习资料（防幻觉：只返回资料库里真实存在的内容）
# --------------------------------------------------------------------------- #
@tool
def search_learning_resources(query: str) -> str:
    """在本地学习资料库中检索与 query 相关的真实资料（BM25 关键词检索）。

    当你需要为学生的薄弱点找具体学习材料时调用。
    只会返回资料库里真实存在的条目；若返回"未找到"，说明资料库没有相关内容，
    此时必须如实告知学生，禁止编造资料名或链接。
    """
    docs = retrieve(query, k=3)
    if not docs:
        return "（资料库中未找到匹配资料——请勿编造，建议如实告知学生该主题暂无现成材料）"
    lines = []
    for d in docs:
        lines.append(
            f"- [{d.metadata.get('type', '')}] {d.metadata.get('title', '')} "
            f"({d.metadata.get('url', '')})：{_clip(d.page_content, _MAX_SNIPPET)}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 工具 2：读取长期画像（这个学生是谁）
# --------------------------------------------------------------------------- #
@tool
def get_student_profile(user_id: str) -> str:
    """读取该学生的长期画像（6 维：知识基础 / 学习目标 / 认知风格 / 薄弱点 / 资源偏好 / 学习时间）。"""
    profile = db.load_profile(user_id)
    if profile is None:
        return "（该学生暂无画像记录，可能是首次使用）"
    return _clip(profile.model_dump_json(ensure_ascii=False), _MAX_JSON)


# --------------------------------------------------------------------------- #
# 工具 3：读取最近一次学情（这个学生上次学得怎么样）
# --------------------------------------------------------------------------- #
@tool
def get_last_session(user_id: str) -> str:
    """读取该学生最近一次学习会话的学情快照（掌握度 / 亮点 / 薄弱项 / 当时的建议）。"""
    session = db.load_last_session(user_id)
    if session is None:
        return "（该学生为首次学习，没有历史学情，不要编造「上次」的数据）"
    return (
        f"时间：{session.created_at or '未记录'}；"
        f"掌握度：{session.mastery or '未记录'}；"
        f"亮点：{'、'.join(session.strengths) or '无'}；"
        f"薄弱项：{'、'.join(session.gaps) or '无'}；"
        f"当时建议：{'、'.join(session.suggestions) or '无'}"
    )


# --------------------------------------------------------------------------- #
# 工具 4：统计历史会话数（判断新老学员）
# --------------------------------------------------------------------------- #
@tool
def count_history_sessions(user_id: str) -> str:
    """统计该学生历史学习会话次数（0 表示新学员）。用来判断是否需要做纵向对比。"""
    return f"该学生历史学习会话数：{db.count_sessions(user_id)}"


# 供 Agent 绑定的一组工具
TUTOR_TOOLS = [
    search_learning_resources,
    get_student_profile,
    get_last_session,
    count_history_sessions,
]

# 工具名 → 可调用对象，供执行器按名分发
TOOL_REGISTRY = {t.name: t for t in TUTOR_TOOLS}

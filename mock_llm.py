"""Mock LLM 层 —— 让 Demo 在没有任何 API Key 的情况下也能跑通完整多智能体流程。

原理：
- 真实项目里每个 Agent 都通过 `app.llm.get_structured_model(Schema)` 拿到一个
  `.invoke(messages) -> Schema 实例` 的对象；
- 本模块把 `get_structured_model` 替换成一个「假模型」，根据传入的 Schema 类型
  返回一份**写死的、结构合法**的样例输出。
- ProfileAgent 里的「姓名/专业正则提取」是真实代码，会基于用户输入覆盖 mock 结果，
  所以 demo 里你填的名字/专业是真的会显示的。
- ResourceAgent 调用的 BM25 检索是真实代码，链接全部来自本地 resources.json 真实数据，
  因此 demo 中展示的「只推荐真实链接、不编造」恰好是 RAG 防幻觉机制的真实体现。

部署到 HuggingFace Spaces 时只要不填 LLM_API_KEY，就会自动走这个 mock 模式，
零配置、零费用即可展示完整 5 智能体流程。填了 Key 则自动切换为真实 DeepSeek。
"""
from __future__ import annotations

import json
import os

from app.models import (
    Dimension,
    Plan,
    PlanStep,
    Profile,
    Quiz,
    QuizQuestion,
    ResourceItem,
    ResourceList,
    Review,
)

# --------------------------------------------------------------------------- #
# 真实资料库（用于让 mock 资源推荐也返回真实链接，体现防幻觉）
# --------------------------------------------------------------------------- #
_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "app", "data", "resources.json")


def _load_real_resources() -> list[dict]:
    try:
        with open(_CORPUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


_REAL = _load_real_resources()


def _sample_resources() -> ResourceList:
    picks = [
        ("FastAPI 官方教程", "https://fastapi.tiangolo.com/zh/", "doc"),
        ("LangGraph 多智能体编排入门", "https://langchain-ai.github.io/langgraph/", "doc"),
        ("RAG 检索增强生成原理与实践", "https://python.langchain.com.cn/docs/modules/data_connection/", "article"),
        ("Docker 容器化 Python 应用", "https://docs.docker.com/get-started/", "doc"),
    ]
    items: list[ResourceItem] = []
    for title, url, rtype in picks:
        meta = next((r for r in _REAL if r.get("title") == title), None)
        desc = meta.get("content", "")[:60] + "…" if meta else ""
        items.append(
            ResourceItem(
                title=title,
                url=url,
                type=rtype,
                description=desc,
                relevance="与学习计划中的对应步骤直接匹配，且为官方/权威资料。",
            )
        )
    return ResourceList(items=items)


# --------------------------------------------------------------------------- #
# 各 Agent 的样例输出
# --------------------------------------------------------------------------- #
def _sample_profile() -> Profile:
    return Profile(
        name=None,  # 由 ProfileAgent 正则从用户输入提取并覆盖
        major=None,
        knowledgeBase=Dimension(level="intermediate", details="Python 基础一般，能写简单脚本"),
        learningGoal=Dimension(level="clear", details="想做 AI Agent 方向，目标是能独立交付一个 Agent 项目"),
        cognitiveStyle=Dimension(level="逻辑型", details="偏好先理解原理再动手"),
        weakPoints=Dimension(level="high", details="算法与数据结构薄弱"),
        resourcePreference=Dimension(level="视频", details="偏好视频 + 动手项目"),
        studyTime=Dimension(level="medium", details="每周约 10 小时"),
        summary="",
    )


def _sample_plan() -> Plan:
    steps = [
        PlanStep(step=1, title="Python 与 FastAPI 基础巩固", description="补齐类型注解、Pydantic、异步视图，能独立写一个小 API。", est_minutes=180),
        PlanStep(step=2, title="LangChain / LangGraph 多智能体入门", description="理解 StateGraph 如何把多个 Agent 串成有状态工作流。", est_minutes=240),
        PlanStep(step=3, title="RAG 检索增强实践", description="用 BM25 接入本地资料库，理解防幻觉机制。", est_minutes=200),
        PlanStep(step=4, title="动手做一个校园场景 Agent", description="把前几步串起来，做一个能跑的最小可用 Agent。", est_minutes=300),
        PlanStep(step=5, title="部署与工程化（Docker）", description="容器化并部署到 HuggingFace / Render，形成作品集。", est_minutes=150),
    ]
    return Plan(goal="从零构建并部署第一个 AI Agent 项目", steps=steps, total_minutes=sum(s.est_minutes for s in steps))


def _sample_quiz() -> Quiz:
    return Quiz(
        questions=[
            QuizQuestion(
                q="LangGraph 中用什么把多个智能体串成有状态工作流？",
                options=["StateGraph", "普通函数调用", "SQL 触发器", "CSS 动画"],
                answer="StateGraph",
                explanation="StateGraph 用节点+边定义多智能体流程，支持状态持久化与条件分支。",
            ),
            QuizQuestion(
                q="RAG 主要通过什么手段缓解大模型幻觉？",
                options=["把相关资料检索后拼进提示", "增大模型参数量", "更换 GPU", "减少训练数据"],
                answer="把相关资料检索后拼进提示",
                explanation="RAG 用检索到的真实资料作为事实依据约束生成，降低编造风险。",
            ),
            QuizQuestion(
                q="FastAPI 依靠哪个库实现请求体类型校验与自动文档？",
                options=["Pydantic", "NumPy", "Pandas", "Requests"],
                answer="Pydantic",
                explanation="FastAPI 基于 Pydantic 做数据校验，并据此自动生成 OpenAPI 文档。",
            ),
        ]
    )


def _sample_review() -> Review:
    return Review(
        mastery="基础入门、方向明确，已具备构建第一个 Agent 的知识框架",
        strengths=["目标清晰，方向聚焦 AI Agent", "已具备 Python 基础", "学习意愿强、每周时间充足"],
        gaps=["算法与数据结构需补齐", "工程化部署经验不足"],
        suggestions=["每天 1-2 道 LeetCode 打底", "先完整跑通本项目再扩展功能", "把项目部署到 HuggingFace 形成可点开的简历作品集"],
    )


_SAMPLES = {
    "Profile": _sample_profile,
    "Plan": _sample_plan,
    "ResourceList": _sample_resources,
    "Quiz": _sample_quiz,
    "Review": _sample_review,
}


class _FakeStructured:
    """替代 get_structured_model 返回的对象：invoke() 直接给出写死的样例。"""

    def __init__(self, schema):
        self._schema = schema

    def invoke(self, messages=None):
        factory = _SAMPLES.get(self._schema.__name__)
        if factory is None:
            raise RuntimeError(f"mock 未覆盖 schema: {self._schema.__name__}")
        return factory()


def install():
    """猴子补丁：把 app.llm.get_structured_model 换成假模型工厂。"""
    import app.llm as llm_mod

    def _fake_get_structured_model(schema, temperature=None, max_tokens=None):
        return _FakeStructured(schema)

    llm_mod.get_structured_model = _fake_get_structured_model
    # 兜底：万一有地方直接调用 get_chat_model
    def _fake_get_chat_model(temperature=None, max_tokens=None):
        raise RuntimeError("mock 模式下不应调用真实 ChatModel；请检查调用路径。")
    llm_mod.get_chat_model = _fake_get_chat_model

"""端到端多智能体管线测试。

把 5 个智能体的 LLM 全部 mock 掉，跑完整张 LangGraph，
验证「画像 → 计划 → 资源(RAG) → 测验 → 复盘」能产出全部产物。
注意：ResourceAgent 的 BM25 检索走真实本地语料，不依赖网络。
"""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

from app.graph import build_graph
from app.models import (
    Dimension,
    Message,
    Plan,
    PlanStep,
    Profile,
    Quiz,
    QuizQuestion,
    ResourceItem,
    ResourceList,
    Review,
)


def _fake_profile() -> Profile:
    return Profile(
        knowledgeBase=Dimension(level="beginner", details="Python 基础一般"),
        learningGoal=Dimension(level="clear", details="想做 AI Agent"),
        cognitiveStyle=Dimension(level="visual", details="偏好视频/图示"),
        weakPoints=Dimension(level="high", details="算法薄弱"),
        resourcePreference=Dimension(level="video", details="爱看视频"),
        studyTime=Dimension(level="medium", details="每周约 10 小时"),
        summary="",
    )


def _fake_plan() -> Plan:
    return Plan(
        goal="四周入门 AI Agent",
        steps=[
            PlanStep(step=1, title="补齐 Python 基础", description="变量/函数/面向对象", est_minutes=180),
            PlanStep(step=2, title="学习 LangGraph", description="用 StateGraph 编排多智能体", est_minutes=240),
        ],
        total_minutes=420,
    )


def _fake_resources() -> ResourceList:
    return ResourceList(
        items=[
            ResourceItem(
                title="LangGraph 多智能体编排入门",
                url="https://langchain-ai.github.io/langgraph/",
                type="doc",
                description="官方文档",
                relevance="直接对应学习计划第 2 步",
            )
        ]
    )


def _fake_quiz() -> Quiz:
    return Quiz(
        questions=[
            QuizQuestion(
                q="LangGraph 中用于定义多智能体工作流的类是？",
                options=["StateGraph", "DataFrame", "Router"],
                answer="StateGraph",
                explanation="LangGraph 用 StateGraph 把节点串成有状态图。",
            )
        ]
    )


def _fake_review() -> Review:
    return Review(
        mastery="入门阶段",
        strengths=["目标清晰", "有 Python 基础"],
        gaps=["算法薄弱"],
        suggestions=["每天刷 1 道 LeetCode", "先跑通一个 LangGraph Demo"],
    )


@contextlib.contextmanager
def _mock_all_llms():
    mapping = {
        "app.agents.profile_agent.get_structured_model": _fake_profile(),
        "app.agents.planner_agent.get_structured_model": _fake_plan(),
        "app.agents.resource_agent.get_structured_model": _fake_resources(),
        "app.agents.quiz_agent.get_structured_model": _fake_quiz(),
        "app.agents.review_agent.get_structured_model": _fake_review(),
    }
    with contextlib.ExitStack() as stack:
        for target, retval in mapping.items():
            m = MagicMock()
            m.return_value.invoke.return_value = retval
            stack.enter_context(patch(target, m))
        yield


def test_full_pipeline_produces_all_artifacts():
    with _mock_all_llms():
        graph = build_graph()
        state = {
            "user_id": "student-001",
            "messages": [
                Message(role="user", content="我叫杨运栋，软件工程专业，Python 基础一般，想做 AI Agent")
            ],
            "resources": [],
            "errors": [],
        }
        final = graph.invoke(state)

    assert final["profile"] is not None
    assert final["profile"].name == "杨运栋"
    assert final["profile"].major == "软件工程"

    assert final["plan"] is not None
    assert len(final["plan"].steps) == 2
    assert final["plan"].total_minutes == 420

    assert isinstance(final["resources"], list)
    assert len(final["resources"]) >= 1

    assert final["quiz"] is not None
    assert len(final["quiz"].questions) == 1

    assert final["review"] is not None
    assert final["review"].suggestions

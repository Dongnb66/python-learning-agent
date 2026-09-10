"""端到端多智能体管线测试。

把 LLM 全部 mock 掉，跑完整张 LangGraph，
验证「画像 → 计划 → 资源(RAG) → 测验 → 复盘 →（条件边）自主辅导」能产出全部产物。
注意：
- ResourceAgent 的 BM25 检索走真实本地语料，不依赖网络；
- 辅导节点强制 TUTOR_MODE=policy（规则策略），避免单测触发真实模型调用；
- 数据库切到临时文件，不污染项目库。
"""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from app import db
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


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    db.init_db()
    return engine


@pytest.fixture()
def policy_mode(monkeypatch):
    monkeypatch.setenv("TUTOR_MODE", "policy")


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


def test_full_pipeline_produces_all_artifacts(tmp_db, policy_mode):
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

    # 复盘给出了薄弱项（算法薄弱）→ 条件边应走到自主辅导节点
    assert final.get("tutoring") is not None, "有薄弱项时必须触发自主辅导"
    assert final["tutoring"].tools_used, "辅导节点必须真实调用工具"
    assert final["tutoring"].trace, "必须留下可审计的决策轨迹"
    assert final.get("agent_trace") == final["tutoring"].trace

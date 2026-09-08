"""ProfileAgent 测试。

关键设计：用 unittest.mock 替换 LLM 调用，使测试无需真实 API Key 即可运行，
验证「画像抽取 + 姓名/专业正则覆盖」两条主路径。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents.profile_agent import (
    build_profile_node,
    extract_major,
    extract_name,
)
from app.models import Dimension, Message, Profile


# --------------------------------------------------------------------------- #
# 1. 纯函数：正则提取
# --------------------------------------------------------------------------- #
def test_extract_name():
    assert extract_name("老师好，我叫杨运栋，来自张家界") == "杨运栋"
    assert extract_name("我是陈晨") == "陈晨"          # 「我是」句式
    assert extract_name("我的名字是李雷") == "李雷"     # 「我的名字是」句式
    # 黑名单 / 非人名应被过滤
    assert extract_name("我是小明") is None           # 小明在黑名单
    assert extract_name("我叫大家") is None
    assert extract_name("没有任何名字信息") is None


def test_extract_major():
    assert extract_major("我是计算机科学与技术专业的学生") == "计算机科学与技术"
    assert extract_major("我在软件学院读书") == "软件"
    assert extract_major("我学的是网络安全方向") == "网络安全"
    assert extract_major("他是电子信息系的") == "电子信息"
    # 否定 / 疑问表述应被过滤
    assert extract_major("没有任何专业信息") is None
    assert extract_major("我不记得是什么专业") is None


# --------------------------------------------------------------------------- #
# 2. 节点集成：mock LLM，验证节点返回与正则覆盖
# --------------------------------------------------------------------------- #
def _fake_profile() -> Profile:
    return Profile(
        knowledgeBase=Dimension(level="beginner", details="刚入门"),
        learningGoal=Dimension(level="clear", details="想做 AI Agent"),
        cognitiveStyle=Dimension(level="visual", details="偏好图示"),
        weakPoints=Dimension(level="high", details="算法薄弱"),
        resourcePreference=Dimension(level="video", details="爱看视频"),
        studyTime=Dimension(level="medium", details="每周 10 小时"),
        summary="",
    )


@patch("app.agents.profile_agent.get_structured_model")
def test_build_profile_node_applies_regex(mock_get_structured_model):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = _fake_profile()
    mock_get_structured_model.return_value = fake_llm

    state = {
        "user_id": "u1",
        "messages": [
            Message(role="user", content="我叫杨运栋，我是软件工程专业的学生，想学 AI Agent。")
        ],
        "resources": [],
        "errors": [],
    }
    out = build_profile_node(state)
    profile = out["profile"]
    assert isinstance(profile, Profile)
    # 正则高置信覆盖 LLM 结果
    assert profile.name == "杨运栋"
    assert profile.major == "软件工程"
    # 自动生成总结
    assert profile.summary


# --------------------------------------------------------------------------- #
# 3. 应用冒烟测试：/health 与图可编译
# --------------------------------------------------------------------------- #
def test_health_endpoint():
    from app.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_graph_compiles():
    from app.graph import build_graph

    g = build_graph()
    assert g is not None

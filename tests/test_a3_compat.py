"""A3 前端兼容 API 层测试。

验证 /api 下的 A3 兼容端点与 A3 React 用户端的契约一致：
- 响应统一为 {success, data} 信封，错误为 {success, error} + HTTP 状态码；
- /api/generate 跑完整 LangGraph 管线并产出 A3 资源包结构；
- /api/tutor/ask 走 ReAct 辅导智能体（policy 模式）且返回可审计轨迹；
- 前后测判分在服务端完成（下发题目不含答案）。

LLM 全部替换为样例输出（同 test_pipeline 的做法），不依赖网络与 Key。
"""
from __future__ import annotations

import contextlib
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import a3_store, db
from app.efficacy_bank import get_questions
from tests.test_pipeline import (
    _fake_plan,
    _fake_profile,
    _fake_quiz,
    _fake_resources,
    _fake_review,
)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """隔离环境：临时 DB + 临时 JSON 存储 + policy 辅导 + mock LLM。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'a3_compat_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    monkeypatch.setenv("A3_STORE_PATH", str(tmp_path / "a3_store.json"))
    monkeypatch.setenv("TUTOR_MODE", "policy")
    db.init_db()
    a3_store.save(a3_store.load())  # 初始化空存储文件

    from app.main import app

    with TestClient(app) as c:
        yield c


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


DIALOGUE = [
    {"role": "user", "content": "我叫杨运栋，软件工程专业"},
    {"role": "assistant", "content": "好的，了解了你的专业。你的学习目标是什么？"},
    {"role": "user", "content": "想系统学习数据结构与算法，每周能投入 10 小时"},
]


# --------------------------------------------------------------------------- #
# 信封与健康检查
# --------------------------------------------------------------------------- #
def test_a3_health_envelope(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"


def test_error_shape_is_a3_style(client):
    resp = client.get("/api/resource/not-exist-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert "error" in body


# --------------------------------------------------------------------------- #
# 画像对话与生成
# --------------------------------------------------------------------------- #
def test_profile_chat_returns_reply(client):
    resp = client.post(
        "/api/profile/chat",
        json={"messages": [{"role": "user", "content": "我是计算机专业的学生"}]},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["reply"]
    assert isinstance(data["latency"], int)


def test_profile_chat_empty_messages_rejected(client):
    resp = client.post("/api/profile/chat", json={"messages": []})
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_profile_generate_shape(client):
    with _mock_all_llms():
        resp = client.post("/api/profile/generate", json={"dialogueHistory": DIALOGUE})
    assert resp.status_code == 200
    data = resp.json()["data"]
    profile = data["profile"]
    assert profile["id"].startswith("profile-")
    assert profile["studentName"] == "杨运栋"
    assert set(profile["dimensions"].keys()) == {
        "knowledgeBase",
        "learningGoal",
        "cognitiveStyle",
        "weakPoints",
        "resourcePreference",
        "studyTime",
    }
    assert data["agentResult"]["agent"] == "ProfileAgent"
    return profile["id"]


def test_profile_get_and_list(client):
    pid = test_profile_generate_shape(client)

    resp = client.get(f"/api/profile/{pid}")
    assert resp.status_code == 200
    assert resp.json()["data"]["profile"]["id"] == pid

    resp = client.get("/api/profile")
    assert resp.status_code == 200
    assert any(p["id"] == pid for p in resp.json()["data"]["profiles"])

    resp = client.get("/api/profile/profile-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 资源生成（完整管线 → A3 资源包）
# --------------------------------------------------------------------------- #
def test_generate_produces_a3_package(client):
    with _mock_all_llms():
        resp = client.post(
            "/api/generate",
            json={"dialogueHistory": DIALOGUE, "course": "数据结构", "topic": "二叉树遍历算法"},
        )
    assert resp.status_code == 200
    resource = resp.json()["data"]["resource"]

    # A3 前端 ResourceDetail / Generate 消费的全部字段
    for key in (
        "id",
        "course",
        "topic",
        "studentName",
        "createdAt",
        "lecture",
        "mindMap",
        "quiz",
        "reading",
        "caseStudy",
        "review",
        "knowledgeSources",
        "executionSteps",
        "profile",
    ):
        assert key in resource, f"资源包缺少字段 {key}"

    assert resource["course"] == "数据结构"
    assert resource["mindMap"]["name"] == "二叉树遍历算法"
    assert resource["mindMap"]["children"]
    assert resource["lecture"].startswith("# ")
    assert len(resource["executionSteps"]) == 5
    assert {s["agent"] for s in resource["executionSteps"]} == {
        "ProfileAgent",
        "PlannerAgent",
        "ResourceAgent",
        "QuizAgent",
        "ReviewAgent",
    }
    assert resource["review"]["passed"] is True
    assert resource["profile"]["dimensions"]["learningGoal"]["details"]

    # 资源包已持久化：列表与详情可查
    resp = client.get("/api/resource")
    assert any(r["id"] == resource["id"] for r in resp.json()["data"]["resources"])
    resp = client.get(f"/api/resource/{resource['id']}")
    assert resp.status_code == 200
    return resource


def test_generate_infers_course_from_major(client):
    with _mock_all_llms():
        resp = client.post("/api/generate", json={"dialogueHistory": DIALOGUE})
    assert resp.status_code == 200
    resource = resp.json()["data"]["resource"]
    # 软件工程 → 数据结构（推断链路）
    assert resource["course"] == "数据结构"
    assert resource["topic"] == "二叉树遍历算法"


# --------------------------------------------------------------------------- #
# 学习路径
# --------------------------------------------------------------------------- #
def test_path_from_profile(client):
    with _mock_all_llms():
        client.post("/api/profile/generate", json={"dialogueHistory": DIALOGUE})
        pid = client.get("/api/profile").json()["data"]["profiles"][0]["id"]
        resp = client.get(f"/api/path/{pid}")
    assert resp.status_code == 200
    path = resp.json()["data"]["path"]
    assert path["stages"]
    stage = path["stages"][0]
    for key in ("stage", "title", "estimatedTime", "objective", "topics", "resources"):
        assert key in stage


def test_path_missing_profile_404(client):
    resp = client.get("/api/path/profile-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 智能辅导（ReAct）
# --------------------------------------------------------------------------- #
def test_tutor_ask_runs_react_loop(client):
    resp = client.post(
        "/api/tutor/ask",
        json={"question": "我算法总是学不进去怎么办？", "profileId": "stu-demo"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["answer"]
    assert data["hasAnswer"] is True
    assert data["toolsUsed"], "policy 模式也必须真实调用工具"
    assert data["trace"]
    assert data["trace"][0]["round"] == 1
    assert data["grounded"] is True


def test_tutor_ask_empty_question_rejected(client):
    resp = client.post("/api/tutor/ask", json={"question": "  "})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# 模型管理
# --------------------------------------------------------------------------- #
def test_models_list_add_delete(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["builtIn"]
    assert body["custom"] == []

    resp = client.post(
        "/api/models",
        json={"id": "Qwen Local", "name": "本地 Qwen", "baseUrl": "http://localhost:11434/v1", "apiKey": "k", "model": "qwen2"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "qwen-local"  # id 规范化

    body = client.get("/api/models").json()["data"]
    assert any(m["id"] == "qwen-local" for m in body["custom"])

    resp = client.delete("/api/models/qwen-local")
    assert resp.status_code == 200
    body = client.get("/api/models").json()["data"]
    assert not any(m["id"] == "qwen-local" for m in body["custom"])


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
def test_knowledge_crud(client):
    resp = client.post(
        "/api/knowledge/upload",
        json={"title": "机器学习基础", "content": "# 概述\n监督学习是……\n## 过拟合\n正则化可缓解过拟合。"},
    )
    assert resp.status_code == 200
    doc = resp.json()["data"]["document"]
    assert doc["id"]
    assert doc["sectionCount"] == 2

    resp = client.get("/api/knowledge")
    docs = resp.json()["data"]["documents"]
    assert any(d["id"] == doc["id"] for d in docs)

    resp = client.put(
        f"/api/knowledge/{doc['id']}",
        json={"title": "机器学习基础（修订）", "content": "# 概述\n更新后的内容。"},
    )
    assert resp.status_code == 200
    updated = resp.json()["data"]["document"]
    assert updated["title"] == "机器学习基础（修订）"
    assert updated["sectionCount"] == 1

    assert client.get("/api/knowledge/doc-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
# 前后测效果实证（服务端判分，下发无答案）
# --------------------------------------------------------------------------- #
def test_efficacy_questions_hide_answers(client):
    resp = client.get("/api/efficacy/questions?topic=机器学习")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 5
    for q in data["questions"]:
        assert "answer" not in q
        assert "explanation" not in q
        assert q["options"]


def test_efficacy_submit_scores_server_side(client):
    correct_answers = {q["id"]: q["answer"] for q in get_questions("机器学习")}
    resp = client.post(
        "/api/efficacy/submit",
        json={
            "studentId": "stu-42",
            "studentName": "杨运栋",
            "topic": "机器学习",
            "phase": "pre",
            "answers": correct_answers,
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["score"] == 100
    assert data["correct"] == 5
    assert data["detail"][0]["correct"] is True
    assert data["record"]["phase"] == "pre"

    # 同一学生 post 阶段 → 计算提升
    resp = client.post(
        "/api/efficacy/submit",
        json={
            "studentId": "stu-42",
            "studentName": "杨运栋",
            "topic": "机器学习",
            "phase": "post",
            "answers": correct_answers,
        },
    )
    data = resp.json()["data"]
    assert data["improvement"] == 0  # pre=100, post=100


def test_efficacy_submit_rejects_bad_phase(client):
    resp = client.post(
        "/api/efficacy/submit",
        json={"topic": "机器学习", "phase": "mid", "answers": {}},
    )
    assert resp.status_code == 400


def test_efficacy_summary_merges_seeds_and_real(client):
    resp = client.get("/api/efficacy/summary")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["totalStudents"] >= 5  # 至少包含种子样例
    assert data["avgPost"] > data["avgPre"]
    assert data["avgImprovement"] > 0
    assert data["perTopic"]
    assert any(s["seeded"] for s in data["sampleStudents"])


# --------------------------------------------------------------------------- #
# 学习分析
# --------------------------------------------------------------------------- #
def test_analytics_counts(client):
    with _mock_all_llms():
        client.post("/api/generate", json={"dialogueHistory": DIALOGUE})
    client.post(
        "/api/knowledge/upload",
        json={"title": "t", "content": "c"},
    )

    resp = client.get("/api/analytics")
    assert resp.status_code == 200
    a = resp.json()["data"]["analytics"]
    assert a["totalProfiles"] >= 1
    assert a["totalResources"] >= 1
    assert a["totalDocs"] >= 1
    assert a["totalLogs"] >= 1
    assert "generate" in a["logsByType"]
    assert a["avgLatency"] >= 0
    assert a["recentResources"]
    # Analytics.jsx 会读 recentLogs.length，缺字段会导致整页白屏
    assert "recentLogs" in a

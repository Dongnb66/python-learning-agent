"""/api/metrics 与 /api/traces 的契约测试。

为什么单独测这两个端点
----------------------
可观测性的价值全在「线上真能查到」——本地埋点对了，端点漏字段、trace 查不回来、
Prometheus 格式不合法，都等于没做。这里以真实请求（TestClient 走完整管线）
验证三件事：

1. 每次业务请求都能拿到 `X-Trace-Id`，且**按 id 能查回**这次请求的链路；
2. 链路里确实有 8 个节点 span 与模型/工具调用 span，耗时字段可用；
3. 指标端点能给出防幻觉守卫的计数（`guardrails`），且真实/模拟模型调用分开。

LLM 全部替换为样例输出；辅导节点走 policy 模式 —— 不访问网络、不依赖 Key。
"""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app import db, telemetry
from tests.test_pipeline import (
    _fake_plan,
    _fake_profile,
    _fake_quiz,
    _fake_resources,
    _fake_review,
)

# 期望被走到 / 可能被跳过的图节点
_ALWAYS_NODES = {
    "node.load_memory",
    "node.profile",
    "node.planner",
    "node.resource",
    "node.quiz",
    "node.review",
    "node.save_memory",
}

DIALOGUE = [
    {"role": "user", "content": "我叫张三，计算机科学与技术专业"},
    {"role": "user", "content": "想学 Python 和 FastAPI，算法比较弱，每周能学 10 小时"},
]


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


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """隔离环境：临时 DB + policy 辅导 + 指标清零。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'metrics_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    monkeypatch.setenv("TUTOR_MODE", "policy")
    monkeypatch.delenv("MOCK_LLM", raising=False)  # 明确断言 mock 标志位，不受外部环境干扰
    db.init_db()
    telemetry.reset()

    from app.main import app

    with TestClient(app) as c:
        yield c
    telemetry.reset()


def _learn(client) -> tuple[dict, str]:
    """用样例输出替换 5 个结构化模型（确定性最强，覆盖各端点契约）。"""
    with _mock_all_llms():
        r = client.post("/api/learn", json={"user_id": "metrics-user", "messages": DIALOGUE})
    assert r.status_code == 200, r.text
    return r.json(), r.headers.get("X-Trace-Id", "")


def _learn_via_mock_models(client) -> tuple[dict, str]:
    """不替换工厂函数，走 `MOCK_LLM=1` 的内置假模型路径。

    这条路径**经过** `app.llm` 的计数代理，因此才能验证埋点真的覆盖了
    5 个结构化模型节点 —— 用 MagicMock 顶掉工厂函数会把埋点一起顶掉，
    那样测的是「我有没有 mock 成功」，不是「监控有没有生效」。
    """
    r = client.post("/api/learn", json={"user_id": "metrics-user", "messages": DIALOGUE})
    assert r.status_code == 200, r.text
    return r.json(), r.headers.get("X-Trace-Id", "")


@pytest.fixture()
def mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


# --------------------------------------------------------------------------- #
# 链路
# --------------------------------------------------------------------------- #
def test_learn_returns_trace_id_header(client) -> None:
    _, trace_id = _learn(client)
    assert trace_id, "每次业务请求都必须能拿到 X-Trace-Id"


def test_trace_contains_all_pipeline_nodes(client) -> None:
    _, trace_id = _learn(client)
    body = client.get(f"/api/traces/{trace_id}").json()["trace"]

    names = {s["name"] for s in body["spans"]}
    assert _ALWAYS_NODES <= names, f"缺少节点 span: {_ALWAYS_NODES - names}"
    assert all(s["status"] == "ok" for s in body["spans"])
    assert all(s["duration_ms"] >= 0 for s in body["spans"])
    assert body["span_count"] == len(body["spans"])
    assert body["status"] == "ok"


def test_trace_spans_are_nested_under_nodes(client, mock_llm) -> None:
    """模型/工具调用 span 的父级应是它所属的节点 —— 有层级才看得出「谁慢」。"""
    _, trace_id = _learn_via_mock_models(client)
    spans = client.get(f"/api/traces/{trace_id}").json()["trace"]["spans"]

    llm_spans = [s for s in spans if s["name"].startswith("llm.")]
    node_names = {s["name"] for s in spans if s["name"].startswith("node.")}
    assert llm_spans, "结构化模型调用应产生 llm.* span"
    assert all(s.get("parent") in node_names for s in llm_spans), llm_spans


def test_trace_reports_tool_calls_from_autonomous_agent(client) -> None:
    """自主辅导 Agent 在 policy 模式下也会真实执行工具 —— 计数必须 > 0。"""
    _, trace_id = _learn(client)
    trace = client.get(f"/api/traces/{trace_id}").json()["trace"]

    assert "node.tutor" in {s["name"] for s in trace["spans"]}
    assert trace["counters"].get(telemetry.C_TOOL_CALLS, 0) > 0
    assert any(s["name"].startswith("tool.") for s in trace["spans"])


def test_unknown_trace_returns_404(client) -> None:
    r = client.get("/api/traces/definitely-not-a-trace")
    assert r.status_code == 404
    assert "trace" in r.json()["detail"]


def test_traces_list_respects_limit(client) -> None:
    _learn(client)
    _learn(client)
    ids = client.get("/api/traces?limit=1").json()["trace_ids"]
    assert len(ids) == 1


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def test_metrics_counts_requests_and_splits_mock_calls(client) -> None:
    _learn(client)
    m = client.get("/api/metrics").json()["metrics"]

    assert m["counters"][telemetry.C_REQUESTS] == 1
    calls = m["llm_calls"]
    assert calls["total"] == calls["mock"] + calls["real"]
    # 本用例把工厂函数整个换掉了，所以模型调用计数为 0（埋点被一起顶掉）
    assert calls["total"] == 0
    assert m["mock_llm"] is False, "MOCK_LLM 未设置，标志位不应误报为 mock"


def test_metrics_counts_every_structured_model_call_in_stub_mode(client, mock_llm) -> None:
    """走内置假模型时，5 个结构化模型节点每次请求各调一次模型 —— 计数必须对得上。"""
    _learn_via_mock_models(client)
    m = client.get("/api/metrics").json()["metrics"]

    assert m["mock_llm"] is True
    calls = m["llm_calls"]
    assert calls["mock"] >= 5, calls
    assert calls["real"] == 0  # 全程无真实 API 调用


def test_metrics_exposes_node_timings(client) -> None:
    _learn(client)
    spans = client.get("/api/metrics").json()["metrics"]["spans"]
    for node in _ALWAYS_NODES:
        assert node in spans, f"指标里缺少 {node} 的耗时统计"
        assert spans[node]["count"] == 1
        assert spans[node]["avg_ms"] >= 0


def test_metrics_guardrails_block_is_queryable(client) -> None:
    """防幻觉三件套必须有专门的可查字段（面试时可直接指给人看）。"""
    _learn(client)
    g = client.get("/api/metrics").json()["metrics"]["guardrails"]
    assert set(g) == {
        "retrieval_empty_total",
        "refusals_total",
        "fabricated_links_blocked_total",
    }
    assert g["fabricated_links_blocked_total"] == 0


def test_metrics_prometheus_format(client) -> None:
    _learn(client)
    r = client.get("/api/metrics?format=prometheus")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "learning_agent_counter" in body
    assert "learning_agent_mock_llm" in body
    assert 'span="node_profile"' in body
    # 有 HELP/TYPE 才是合法 exposition 格式
    assert "# TYPE" in body

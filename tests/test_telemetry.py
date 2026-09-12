"""可观测层自身的测试 —— 监控代码没被测过，它给出的数字就不可信。

分四部分：
1. **埋点原语**：span 计时 / 异常标记 / 嵌套父级、trace 归集、计数器双写；
2. **进程级注册表**：累计、快照、有界 trace 存储、Prometheus 渲染；
3. **守卫计数器**：把「防幻觉三道约束」的拦截动作变成可断言的行为
   —— 这是本文件最重要的一节：声明「我做了防幻觉」是空话，
   「拒答一次这个计数器就 +1」才是证据；
4. **模型调用计数代理**：真实 / 模拟分开统计，且 bind_tools 之后仍计数
   （ReAct 循环那一支不遗漏）。
"""
from __future__ import annotations

import pytest

from app import llm, telemetry
from app.agents.resource_agent import build_resources_node
from app.models import Plan, PlanStep, ResourceItem, ResourceList
from app.rag import retrieve

_REAL_TITLE = "LangGraph 多智能体编排入门"
_REAL_URL = "https://langchain-ai.github.io/langgraph/"
_FAKE_URL = "https://not-in-corpus.example.com/awesome"


@pytest.fixture(autouse=True)
def _clean_metrics():
    """每个用例前后都清空指标，避免用例之间互相污染（指标是进程级全局状态）。"""
    telemetry.reset()
    yield
    telemetry.reset()


# --------------------------------------------------------------------------- #
# 1. 埋点原语
# --------------------------------------------------------------------------- #
def test_span_records_duration_into_registry_without_trace() -> None:
    """没有活动 trace 时也要计时成功（把埋点和 request 生命周期解耦）。"""
    with telemetry.span("node.detached"):
        pass

    spans = telemetry.snapshot()["spans"]
    assert spans["node.detached"]["count"] == 1
    assert spans["node.detached"]["total_ms"] >= 0
    assert spans["node.detached"]["errors"] == 0


def test_span_marks_error_and_reraises() -> None:
    """异常必须被记为 error 后原样抛出 —— 埋点不能吞掉异常。"""
    with pytest.raises(ValueError):
        with telemetry.span("node.boom"):
            raise ValueError("炸了")

    info = telemetry.snapshot()["spans"]["node.boom"]
    assert info["errors"] == 1


def test_nested_span_records_parent() -> None:
    """嵌套 span 要能还原调用层级（节点 → 该节点内的模型调用）。"""
    with telemetry.trace_run() as tr:
        with telemetry.span("node.profile"):
            with telemetry.span("llm.Profile"):
                pass

    inner = next(s for s in tr.spans if s.name == "llm.Profile")
    outer = next(s for s in tr.spans if s.name == "node.profile")
    assert inner.parent == "node.profile"
    assert outer.parent == ""


def test_trace_run_collects_counters_and_is_retrievable() -> None:
    """一次请求的 trace 要能事后按 id 查回来，且带上它自己的计数器。"""
    with telemetry.trace_run() as tr:
        telemetry.bump(telemetry.C_REFUSALS)
        telemetry.bump(telemetry.C_REFUSALS)
        with telemetry.span("node.resource"):
            pass

    assert tr.counters[telemetry.C_REFUSALS] == 2
    stored = telemetry.get_trace(tr.trace_id)
    assert stored is not None
    assert stored["trace_id"] == tr.trace_id
    assert stored["counters"][telemetry.C_REFUSALS] == 2
    assert [s["name"] for s in stored["spans"]] == ["node.resource"]


def test_bump_without_active_trace_still_lands_in_registry() -> None:
    """没有 trace 时计数器也要落到进程级指标（否则后台任务无法统计）。"""
    telemetry.bump(telemetry.C_TOOL_CALLS, 3)
    assert telemetry.snapshot()["counters"][telemetry.C_TOOL_CALLS] == 3


def test_trace_marks_status_error_on_exception() -> None:
    """整条请求炸掉时，trace 本身要能看出失败与原因。"""
    with pytest.raises(RuntimeError):
        with telemetry.trace_run() as tr:
            raise RuntimeError("管线挂了")

    stored = telemetry.get_trace(tr.trace_id)
    assert stored["status"] == "error"
    assert "RuntimeError" in stored["error"]


# --------------------------------------------------------------------------- #
# 2. 进程级注册表
# --------------------------------------------------------------------------- #
def test_registry_accumulates_across_requests() -> None:
    for _ in range(3):
        with telemetry.trace_run():
            telemetry.bump(telemetry.C_REQUESTS)
    assert telemetry.snapshot()["counters"][telemetry.C_REQUESTS] == 3


def test_latest_trace_ids_are_newest_first() -> None:
    ids = []
    for _ in range(3):
        with telemetry.trace_run() as tr:
            ids.append(tr.trace_id)
    assert telemetry.latest_trace_ids() == list(reversed(ids))


def test_trace_store_is_bounded() -> None:
    """内存里只留最近 N 条 —— 无界增长是慢性泄漏。"""
    keep = telemetry._TRACE_KEEP
    first = ""
    for i in range(keep + 5):
        with telemetry.trace_run() as tr:
            if i == 0:
                first = tr.trace_id
    assert telemetry.get_trace(first) is None, "最旧的 trace 应已被淘汰"
    assert len(telemetry.latest_trace_ids(keep + 10)) == keep


def test_get_trace_returns_none_for_unknown_id() -> None:
    assert telemetry.get_trace("no-such-trace") is None


def test_prometheus_rendering_sanitizes_span_names() -> None:
    """Prometheus 指标名不允许点号，节点名要能被安全转换。"""
    with telemetry.trace_run():
        with telemetry.span("node.load_memory"):
            pass
        telemetry.bump(telemetry.C_FABRICATED_BLOCKED, 2)

    text = telemetry.render_prometheus()
    assert 'learning_agent_counter{name="fabricated_links_blocked_total"} 2' in text
    assert 'span="node_load_memory"' in text
    assert "node.load_memory" not in text, "渲染后不应残留非法字符"
    assert text.endswith("\n")


def test_snapshot_separates_mock_and_real_llm_calls() -> None:
    """真实调用 = 总数 - Mock 数，避免把离线压测数字当成真实 API 调用量。"""
    telemetry.bump(telemetry.C_LLM_CALLS, 7)
    telemetry.bump(telemetry.C_LLM_MOCK_CALLS, 5)
    telemetry.bump(telemetry.C_LLM_ERRORS, 1)

    info = telemetry.snapshot()["llm_calls"]
    assert (info["total"], info["mock"], info["real"], info["errors"]) == (7, 5, 2, 1)


def test_mock_mode_tracks_env(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    assert telemetry.mock_mode() is True
    monkeypatch.setenv("MOCK_LLM", "false")
    assert telemetry.mock_mode() is False


# --------------------------------------------------------------------------- #
# 3. 守卫计数器：防幻觉的运行时证据
# --------------------------------------------------------------------------- #
def _plan(*titles: str) -> Plan:
    return Plan(
        goal="测试用计划",
        steps=[
            PlanStep(step=i, title=t, description="", est_minutes=60)
            for i, t in enumerate(titles, start=1)
        ],
        total_minutes=60 * len(titles),
    )


def _state(plan: Plan | None) -> dict:
    return {"user_id": "telemetry-user", "messages": [], "plan": plan, "errors": []}


def test_guard_counter_bumps_on_empty_retrieval() -> None:
    """第①道：阈值过滤后为空 → retrieval_empty_total 计数（而不是静默返回空）。"""
    assert retrieve("超导量子比特架构") == []
    counters = telemetry.snapshot()["counters"]
    assert counters[telemetry.C_RETRIEVAL_EMPTY] == 1
    assert counters[telemetry.C_RETRIEVAL] == 1


def test_guard_counter_bumps_on_code_level_refusal() -> None:
    """第②道：检索为空导致代码级拒答 → refusals_total 计数。"""
    build_resources_node(_state(_plan("超导量子比特架构", "元代青花瓷钴料鉴定")))
    assert telemetry.snapshot()["guardrails"]["refusals_total"] == 1


def test_guard_counter_counts_blocked_fabricated_links() -> None:
    """第③道：模型编造的条目被白名单拦下 → 计数器要如实反映拦了几条。"""
    from unittest.mock import MagicMock, patch

    items = [
        ResourceItem(title=_REAL_TITLE, url=_REAL_URL, type="doc", description="真实"),
        ResourceItem(title="史上最强速成课", url=_FAKE_URL, type="video", description="幻觉"),
        ResourceItem(title="另一个幻觉", url="https://another-fake.example.com/x", type="doc"),
    ]
    m = MagicMock()
    m.return_value.invoke.return_value = ResourceList(items=items)
    with patch("app.agents.resource_agent.get_structured_model", m):
        out = build_resources_node(_state(_plan("学习 LangGraph")))

    assert [r.url for r in out["resources"]] == [_REAL_URL]
    counters = telemetry.snapshot()["counters"]
    assert counters[telemetry.C_FABRICATED_BLOCKED] == 2
    assert counters[telemetry.C_RESOURCES_RETURNED] == 1


def test_guardrails_block_is_present_and_zero_in_clean_run() -> None:
    """干净输入下三道守卫都不该被触发 —— 计数为 0 才说明「没有幻觉」不是运气。"""
    from unittest.mock import MagicMock, patch

    m = MagicMock()
    m.return_value.invoke.return_value = ResourceList(
        items=[ResourceItem(title=_REAL_TITLE, url=_REAL_URL, type="doc")]
    )
    with telemetry.trace_run() as tr:
        with patch("app.agents.resource_agent.get_structured_model", m):
            out = build_resources_node(_state(_plan("学习 LangGraph")))
        assert out["resources"]

    assert telemetry.snapshot()["guardrails"] == {
        "retrieval_empty_total": 0,
        "refusals_total": 0,
        "fabricated_links_blocked_total": 0,
    }
    assert tr.counters.get(telemetry.C_RESOURCES_RETURNED) == 1


# --------------------------------------------------------------------------- #
# 4. 模型调用计数代理
# --------------------------------------------------------------------------- #
class _Inner:
    """最小的假 runnable：只实现 invoke / bind_tools。"""

    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom
        self.bound = 0

    def invoke(self, *_a, **_k):  # noqa: ANN002, ANN003
        if self.boom:
            raise RuntimeError("模型调用失败")
        return "ok"

    def bind_tools(self, *_a, **_k):  # noqa: ANN002, ANN003
        self.bound += 1
        return self


def test_instrumented_model_counts_invocations() -> None:
    m = llm._instrument(_Inner(), "Profile")
    assert m.invoke() == "ok"
    assert m.invoke() == "ok"
    assert telemetry.snapshot()["counters"][telemetry.C_LLM_CALLS] == 2


def test_instrumented_model_counts_mock_separately() -> None:
    m = llm._instrument(_Inner(), "Profile", mock=True)
    m.invoke()
    calls = telemetry.snapshot()["llm_calls"]
    assert calls["total"] == 1 and calls["mock"] == 1 and calls["real"] == 0


def test_instrumented_model_counts_errors() -> None:
    m = llm._instrument(_Inner(boom=True), "Quiz")
    with pytest.raises(RuntimeError):
        m.invoke()
    assert telemetry.snapshot()["counters"][telemetry.C_LLM_ERRORS] == 1


def test_bind_tools_stays_instrumented() -> None:
    """ReAct 循环走的是 bind_tools 之后的模型 —— 那一支漏了就等于没统计。"""
    m = llm._instrument(_Inner(), "chat")
    bound = m.bind_tools([])
    assert isinstance(bound, llm.InstrumentedModel)
    bound.invoke()
    assert telemetry.snapshot()["counters"][telemetry.C_LLM_CALLS] == 1


def test_instrumented_model_delegates_unknown_attributes() -> None:
    """未拦截的属性必须透传，否则 LangChain 的 Runnable 协议会被打断。"""

    class _WithExtra:
        def invoke(self, *_a, **_k):
            return "ok"

        def with_structured_output(self, schema, method=None):  # noqa: ANN001, ARG002
            return ("structured", method)

    m = llm._instrument(_WithExtra(), "chat")
    assert m.with_structured_output(object, method="function_calling") == (
        "structured",
        "function_calling",
    )

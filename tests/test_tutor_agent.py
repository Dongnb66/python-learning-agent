"""自主辅导智能体（ReAct）测试。

覆盖两类路径：
1. **规则策略路径**（policy）：无 Key / 显式指定时启用，验证「决策 → 调工具 → 观察」
   循环真的执行了工具、轮次正确、且会按学员新老情况跳过无意义的调用；
2. **模型驱动路径**（llm）：用假模型脚本驱动，验证工具调用循环、
   自主停止、轮数上限、异常降级等关键行为——这些是「自主决策」的核心断言。

全部用临时 SQLite，不污染项目库、不访问网络。
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine

from app import db
from app.agents import tutor_agent
from app.agents.memory_agent import build_save_memory_node
from app.agents.tutor_agent import MAX_ROUNDS, build_tutor_node, route_after_review
from app.models import Review


@pytest.fixture()
def tmp_db(monkeypatch, tmp_path):
    """把 db 层 engine 换到临时文件。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'tutor_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    db.init_db()
    return engine


@pytest.fixture()
def policy_mode(monkeypatch):
    """强制规则策略路径，避免单测触发真实模型调用。"""
    monkeypatch.setenv("TUTOR_MODE", "policy")


def _state(user_id: str, gaps: list[str] | None = None) -> dict:
    return {
        "user_id": user_id,
        "messages": [],
        "review": Review(mastery="入门阶段", gaps=gaps if gaps is not None else ["递归不熟"]),
        "errors": [],
    }


# --------------------------------------------------------------------------- #
# 规则策略路径
# --------------------------------------------------------------------------- #
def test_policy_loop_actually_runs_tools(tmp_db, policy_mode) -> None:
    """规则策略不是空壳：必须真的执行工具，并留下轨迹。"""
    out = build_tutor_node(_state("t-new"))

    session = out["tutoring"]
    assert session is not None
    assert session.mode == "policy"
    assert session.tools_used, "必须至少调用过一个工具"
    assert len(session.trace) == session.rounds > 0
    assert session.answer.strip(), "必须给出结论"
    assert out["errors"] == []


def test_policy_skips_history_for_new_student(tmp_db, policy_mode) -> None:
    """新学员（0 次历史）→ 自主跳过「读上次学情」，不做无意义的纵向对比。"""
    out = build_tutor_node(_state("stu-new"))

    used = out["tutoring"].tools_used
    assert "count_history_sessions" in used
    assert "get_last_session" not in used


def test_policy_reads_history_for_returning_student(tmp_db, policy_mode) -> None:
    """老学员（有历史）→ 应主动读取上次学情。"""
    build_save_memory_node(
        {
            "user_id": "stu-old",
            "review": Review(mastery="进阶中", gaps=["工程经验不足"]),
            "plan": None,
            "errors": [],
        }
    )

    out = build_tutor_node(_state("stu-old"))

    used = out["tutoring"].tools_used
    assert "get_last_session" in used, "老学员必须对照历史学情"
    assert "工程经验不足" in out["tutoring"].answer or "进阶中" in out["tutoring"].answer


def test_policy_retrieves_real_material_for_gap(tmp_db, policy_mode) -> None:
    """检索工具应被调用，且 query 用的是具体薄弱点而非空泛词。"""
    seen: dict[str, str] = {}
    real_execute = tutor_agent._execute_tool

    def spy(name, args, user_id):
        if name == "search_learning_resources":
            seen["query"] = args.get("query", "")
        return real_execute(name, args, user_id)

    tutor_agent._execute_tool = spy  # type: ignore[assignment]
    try:
        build_tutor_node(_state("stu-gap", gaps=["动态规划"]))
    finally:
        tutor_agent._execute_tool = real_execute  # type: ignore[assignment]

    assert "动态规划" in seen.get("query", "")


# --------------------------------------------------------------------------- #
# 模型驱动路径（用假模型脚本验证自主决策行为）
# --------------------------------------------------------------------------- #
class _FakeBoundLLM:
    def __init__(self, script: list[AIMessage], counter: dict):
        self._script = script
        self._counter = counter

    def invoke(self, messages):  # noqa: ANN001
        idx = min(self._counter["n"], len(self._script) - 1)
        self._counter["n"] += 1
        return self._script[idx]


class _FakeLLM:
    """按脚本逐次返回预设 AIMessage，用来模拟模型的自主决策序列。"""

    def __init__(self, script: list[AIMessage]):
        self._script = script
        self._counter = {"n": 0}

    def bind_tools(self, tools):  # noqa: ANN001
        return _FakeBoundLLM(self._script, self._counter)


def _patch_llm(monkeypatch, script: list[AIMessage]) -> None:
    """把模型换成脚本化的假模型，并显式声明走 LLM 路径。

    （.env 里可能是占位符 Key，auto 模式会直接落到规则策略，
    因此这里必须显式设 TUTOR_MODE=llm 才能验证模型驱动的循环行为。）
    """
    monkeypatch.setenv("TUTOR_MODE", "llm")
    monkeypatch.setattr("app.llm.get_chat_model", lambda **_: _FakeLLM(script))


def test_llm_react_calls_tool_then_stops_autonomously(tmp_db, monkeypatch) -> None:
    """模型先自主调工具、再自主判断信息已足够而停止——这正是 Agent 与 Workflow 的分界。"""
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_learning_resources",
                    "args": {"query": "递归"},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="你的卡点在递归的边界条件，建议先看检索到的官方教程。"),
    ]
    _patch_llm(monkeypatch, script)

    out = build_tutor_node(_state("stu-llm"))
    session = out["tutoring"]

    assert session.mode == "llm"
    assert session.rounds == 1, "第一轮调工具、第二轮无 tool_calls 即停止"
    assert session.tools_used == ["search_learning_resources"]
    assert len(session.trace) == 1
    assert session.trace[0].round == 1
    assert "递归" in session.answer


def test_llm_react_respects_max_rounds(tmp_db, monkeypatch) -> None:
    """模型若一直不收敛，必须在轮数上限处强制收口，防止成本失控。"""
    loop_call = AIMessage(
        content="",
        tool_calls=[{"name": "count_history_sessions", "args": {}, "id": "cx"}],
    )
    # 脚本只有一条 → 每次 invoke 都返回「还要继续调工具」，模拟模型不收敛
    _patch_llm(monkeypatch, [loop_call])

    out = build_tutor_node(_state("stu-loop"))
    session = out["tutoring"]

    assert session.rounds == MAX_ROUNDS, f"最多 {MAX_ROUNDS} 轮"
    assert len(session.trace) == MAX_ROUNDS, "每轮都留下轨迹"
    assert session.answer.strip(), "达到上限后仍必须给出结论"


def test_tool_executor_forces_server_side_user_id(tmp_db, monkeypatch) -> None:
    """模型传错 user_id 也不能读到别人的数据——服务端状态优先。"""
    build_save_memory_node(
        {
            "user_id": "victim",
            "review": Review(mastery="机密掌握度", gaps=["机密薄弱项"]),
            "plan": None,
            "errors": [],
        }
    )

    script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_last_session", "args": {"user_id": "victim"}, "id": "c1"}
            ],
        ),
        AIMessage(content="结论"),
    ]
    _patch_llm(monkeypatch, script)

    # 发起者其实是 attacker，模型被诱导去查 victim
    out = build_tutor_node(_state("attacker"))
    observation = out["tutoring"].trace[0].observation

    assert "机密掌握度" not in observation, "不得越权读取他人学情"


def test_llm_failure_falls_back_to_policy(tmp_db, monkeypatch) -> None:
    """模型整体异常时降级到规则策略，管线不中断。"""

    def _boom(**_kwargs):
        raise RuntimeError("模型网关 502")

    monkeypatch.setenv("TUTOR_MODE", "llm")
    monkeypatch.setattr("app.llm.get_chat_model", _boom)

    out = build_tutor_node(_state("stu-err"))

    assert out["tutoring"] is not None
    assert out["tutoring"].mode == "policy"
    assert any("tutor 自主循环失败" in e for e in out["errors"])


# --------------------------------------------------------------------------- #
# 条件边
# --------------------------------------------------------------------------- #
def test_router_branches_on_gaps() -> None:
    """有薄弱项 → 进辅导循环；无薄弱项 → 直接收尾。"""
    assert route_after_review(_state("u", gaps=["递归不熟"])) == "tutor"
    assert route_after_review(_state("u", gaps=[])) == "save_memory"
    assert route_after_review({"user_id": "u", "errors": []}) == "save_memory"


# --------------------------------------------------------------------------- #
# Key 可用性识别（避免对着占位符白跑一次 401）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,expected",
    [
        ("", False),
        (None, False),
        ("   ", False),
        ("sk-your-api-key-here", False),
        ("sk-xxxxxxxx", False),
        ("CHANGE_ME", False),
        ("sk-abcdefghijklmnopqrstuvwxyz012345", True),
    ],
)
def test_has_usable_key_detects_placeholders(key, expected) -> None:
    assert tutor_agent._has_usable_key(key) is expected


def test_auto_mode_with_placeholder_key_stays_offline(tmp_db, monkeypatch) -> None:
    """auto 模式 + 占位符 Key → 应直接走规则策略，不发起注定失败的模型调用。"""

    def _must_not_call(**_kwargs):  # pragma: no cover - 被调用即失败
        raise AssertionError("占位符 Key 不应触发真实模型调用")

    monkeypatch.delenv("TUTOR_MODE", raising=False)
    monkeypatch.setattr("app.llm.get_chat_model", _must_not_call)
    monkeypatch.setattr(tutor_agent.get_settings(), "llm_api_key", "sk-your-api-key-here")

    out = build_tutor_node(_state("stu-placeholder"))

    assert out["tutoring"].mode == "policy"
    assert out["errors"] == []

"""评测脚本自身的测试 —— 评测器没被测过，它就是一厢情愿的数字。

分两部分：
1. `check_case` / `summarize` 的纯逻辑单测（给假的响应，验证判定与算数正确）；
2. **端到端**：以 `MOCK_LLM=1` 离线桩跑完整张 LangGraph，跑完 `eval/bad_cases.json`
   全部用例，断言 100% 通过且编造链接为 0。

第 2 部分同时也是「零 Key 也能端到端跑通」这条能力声明的回归测试。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from app import db, eval_suite
from app.graph import build_graph
from app.models import Message

CASES_PATH = "eval/bad_cases.json"
CORPUS_PATH = "app/data/resources.json"


# --------------------------------------------------------------------------- #
# 1. 纯逻辑：判定与汇总
# --------------------------------------------------------------------------- #
def _resp(urls: list[str | None]) -> dict:
    return {
        "profile": {"name": "张三", "major": "计算机科学与技术"},
        "plan": {"goal": "g", "steps": [], "total_minutes": 0},
        "resources": [{"title": "t", "url": u} for u in urls],
        "quiz": {"questions": []},
        "review": {"mastery": "m", "strengths": [], "gaps": [], "suggestions": []},
    }


_WHITELIST = {"https://good.example.com/a", "https://good.example.com/b"}


def test_check_case_passes_for_verifiable_resources() -> None:
    ok, fails, m = eval_suite.check_case(
        _resp(["https://good.example.com/a"]),
        {"assert": {"all_products_present": True, "resource_url_must_be_local": True}},
        _WHITELIST,
    )
    assert ok, fails
    assert m["fabricated_links"] == []
    assert m["resource_count"] == 1


def test_check_case_flags_fabricated_and_missing_urls() -> None:
    ok, fails, m = eval_suite.check_case(
        _resp(["https://evil.example.com/x", None]),
        {"assert": {"resource_url_must_be_local": True}},
        _WHITELIST,
    )
    assert not ok
    assert m["fabricated_links"] == ["https://evil.example.com/x"]
    assert m["unverifiable_links"] == [None]
    assert any("编造链接" in f for f in fails)
    assert any("缺少出处" in f for f in fails)


def test_check_case_flags_missing_products_and_wrong_profile() -> None:
    resp = _resp(["https://good.example.com/a"])
    resp["plan"] = None
    ok, fails, _ = eval_suite.check_case(
        resp,
        {"assert": {"all_products_present": True, "profile_name_equals": "李四"}},
        _WHITELIST,
    )
    assert not ok
    assert any("产物不齐" in f for f in fails)
    assert any("姓名" in f for f in fails)


def test_summarize_computes_anti_hallucination_rate() -> None:
    results = [
        {"pass": True, "metrics": {"resource_count": 3, "fabricated_links": [], "refusal_expected": False, "refused": False}},
        {"pass": False, "metrics": {"resource_count": 1, "fabricated_links": ["bad"], "refusal_expected": True, "refused": False}},
    ]
    s = eval_suite.summarize(results)
    assert s["cases"] == 2 and s["passed"] == 1
    assert s["resource_count"] == 4
    assert s["fabricated_links"] == 1
    assert s["anti_hallucination_pct"] == 75.0  # 3/4 可溯源
    assert s["refusal_eligible_cases"] == 1 and s["refused_cases"] == 0
    assert s["refusal_rate_pct"] == 0.0


# --------------------------------------------------------------------------- #
# 2. 端到端：离线桩跑完整管线 + 全部评测用例
# --------------------------------------------------------------------------- #
@pytest.fixture()
def mock_env(monkeypatch, tmp_path):
    """离线、临时库：MOCK_LLM + 规则策略，不访问网络、不污染项目库。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("TUTOR_MODE", "policy")
    engine = create_engine(f"sqlite:///{tmp_path / 'eval_test.db'}", future=True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    db.init_db()
    return engine


def _graph_invoker():
    """把 LangGraph 整图包成与 /api/learn 同形的调用器。"""
    graph = build_graph()

    def _dump(obj):
        return obj.model_dump() if hasattr(obj, "model_dump") else obj

    def _invoke(user_id: str, messages: list[dict[str, str]]) -> dict:
        state = {
            "user_id": user_id,
            "messages": [Message(role=m["role"], content=m["content"]) for m in messages],
            "resources": [],
            "errors": [],
        }
        final = graph.invoke(state)
        return {
            "user_id": user_id,
            "profile": _dump(final.get("profile")),
            "plan": _dump(final.get("plan")),
            "resources": [_dump(r) for r in (final.get("resources") or [])],
            "quiz": _dump(final.get("quiz")),
            "review": _dump(final.get("review")),
            "tutoring": _dump(final.get("tutoring")),
            "errors": final.get("errors", []),
        }

    return _invoke


def test_eval_suite_passes_all_cases_offline(mock_env) -> None:
    """零 Key、零网络，跑完全部防幻觉用例，且一条编造链接都没有。"""
    _, cases = eval_suite.load_cases(CASES_PATH)
    local_urls = eval_suite.load_local_urls(CORPUS_PATH)
    assert local_urls, "评测依赖本地资料库作为 URL 白名单，读不到就无法判定"

    report = eval_suite.run_cases(_graph_invoker(), cases, local_urls)
    s = report["summary"]

    assert s["passed"] == s["cases"] == len(cases), [
        (r["id"], r.get("fails") or r.get("error")) for r in report["results"] if not r["pass"]
    ]
    assert s["fabricated_links"] == 0
    assert s["anti_hallucination_pct"] == 100.0
    assert s["resource_count"] > 0, "至少要推荐出资源，否则断言形同虚设"


def test_eval_report_is_json_serializable(mock_env, tmp_path) -> None:
    """报告要能真的落盘（CI 会读它）。"""
    import json

    _, cases = eval_suite.load_cases(CASES_PATH)
    local_urls = eval_suite.load_local_urls(CORPUS_PATH)
    report = eval_suite.run_cases(_graph_invoker(), cases, local_urls)

    path = eval_suite.write_report(report, out_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["cases"] == len(cases)

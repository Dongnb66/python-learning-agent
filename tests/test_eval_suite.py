"""评测脚本自身的测试 —— 评测器没被测过，它就是一厢情愿的数字。

分三部分：
1. `check_case` / `summarize` 的纯逻辑单测（给假的响应，验证判定与算数正确）；
2. **端到端**：以 `MOCK_LLM=1` 离线桩跑完整张 LangGraph，跑完 `eval/bad_cases.json`
   全部用例，断言 100% 通过且编造链接为 0；
3. **对抗集必须真的拒答**：`expect_refusal` 断言真的生效（而不是被评测器忽略），
   且 `GraphInvoker` 的计划注入真的把对抗话题送进了检索层。

第 2、3 部分同时也是「零 Key 也能端到端跑通」这条能力声明的回归测试。
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from app import db, eval_suite

CASES_PATH = "eval/bad_cases.json"
ADVERSARIAL_PATH = "eval/adversarial_cases.json"
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
    """把 LangGraph 整图包成与 /api/learn 同形的调用器（含计划注入能力）。"""
    return eval_suite.GraphInvoker()


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


# --------------------------------------------------------------------------- #
# 3. expect_refusal：断言必须真的生效
# --------------------------------------------------------------------------- #
def test_expect_refusal_fails_when_resources_returned() -> None:
    """声明「必须拒答」的用例，一旦推荐了资源就必须判失败。

    这条断言曾经缺失：mock 返回的资源 URL 全部取自真实资料库，
    `resource_url_must_be_local` 查不出问题 → 三道约束被关掉时用例照样通过。
    """
    ok, fails, m = eval_suite.check_case(
        _resp(["https://good.example.com/a"]),
        {"assert": {"may_refuse_if_no_match": True, "expect_refusal": True}},
        _WHITELIST,
    )
    assert not ok
    assert m["refused"] is False
    assert any("期望拒答" in f for f in fails)


def test_expect_refusal_passes_on_empty_resources() -> None:
    ok, fails, m = eval_suite.check_case(
        _resp([]),
        {"assert": {"resource_url_must_be_local": True, "expect_refusal": True}},
        _WHITELIST,
    )
    assert ok, fails
    assert m["refused"] is True


def test_may_refuse_alone_does_not_require_refusal() -> None:
    """`may_refuse_if_no_match` 只表示「允许为空」——不能反过来要求必须为空。"""
    ok, fails, _ = eval_suite.check_case(
        _resp(["https://good.example.com/a"]),
        {"assert": {"may_refuse_if_no_match": True}},
        _WHITELIST,
    )
    assert ok, fails


# --------------------------------------------------------------------------- #
# 4. 对抗集：真的走到拒答分支
# --------------------------------------------------------------------------- #
def test_graph_invoker_injects_case_plan_so_topic_reaches_retrieval(mock_env) -> None:
    """注入的计划要真的生效，否则对抗话题会被固定桩计划挡住。"""
    case = {
        "id": "probe",
        "messages": [{"role": "user", "content": "我想学量子计算"}],
        "plan_titles": ["超导量子比特架构"],
        "plan_goal": "学习量子计算",
    }
    resp = eval_suite.GraphInvoker()("probe-user", case["messages"], case)
    assert resp["plan"]["goal"] == "学习量子计算"
    assert [s["title"] for s in resp["plan"]["steps"]] == ["超导量子比特架构"]
    assert resp["resources"] == [], "资料库没有该话题 → 必须拒答"


def test_adversarial_cases_all_refuse_offline(mock_env) -> None:
    """对抗集 3 例必须全部拒答（且无编造链接）—— 三道约束的端到端回归。"""
    _, cases = eval_suite.load_cases(ADVERSARIAL_PATH)
    local_urls = eval_suite.load_local_urls(CORPUS_PATH)
    assert cases, "对抗集不能为空"
    assert all(c.get("assert", {}).get("expect_refusal") for c in cases), \
        "对抗集的每条都必须声明 expect_refusal，否则测不到拒答机制"

    report = eval_suite.run_cases(_graph_invoker(), cases, local_urls)
    s = report["summary"]
    assert s["passed"] == s["cases"] == len(cases), [
        (r["id"], r.get("fails") or r.get("error")) for r in report["results"] if not r["pass"]
    ]
    assert s["resource_count"] == 0, "对抗话题在资料库里不存在，不该推荐出任何资源"
    assert s["refused_cases"] == s["refusal_eligible_cases"] == len(cases)
    assert s["fabricated_links"] == 0


def test_adversarial_cases_declare_plan_titles() -> None:
    """每条对抗用例都要声明 plan_titles —— 否则注入不了话题，拒答断言会空转。"""
    _, cases = eval_suite.load_cases(ADVERSARIAL_PATH)
    for c in cases:
        assert c.get("plan_titles"), f"{c['id']} 缺少 plan_titles，拒答分支走不到"


def test_graph_invoker_records_a_trace_per_case(mock_env) -> None:
    """复现报告里的耗时统计靠它 —— 每条用例都要留下可读的链路指标。"""
    invoker = _graph_invoker()
    invoker("trace-user", [{"role": "user", "content": "我想学 Python"}], {"id": "T1"})

    assert len(invoker.traces) == 1
    trace = invoker.traces[0]
    assert trace["case_id"] == "T1"
    assert trace["duration_ms"] >= 0
    assert isinstance(trace["slowest"], list)
    assert isinstance(trace["counters"], dict)


# --------------------------------------------------------------------------- #
# 5. 断言有效性：关掉守卫必须变红
# --------------------------------------------------------------------------- #
def test_adversarial_suite_survives_disabling_refusal_alone(mock_env, monkeypatch) -> None:
    """纵深防御：只关「代码级拒答」，白名单仍会拦下全部无出处条目 → 依旧拒答。"""
    monkeypatch.setenv("ABLATE_REFUSAL", "1")
    _, cases = eval_suite.load_cases(ADVERSARIAL_PATH)
    local_urls = eval_suite.load_local_urls(CORPUS_PATH)

    s = eval_suite.run_cases(_graph_invoker(), cases, local_urls)["summary"]
    assert s["passed"] == s["cases"], "单关一道不该漏 —— 白名单是第二道防线"
    assert s["resource_count"] == 0


def test_adversarial_suite_turns_red_when_two_guards_disabled(mock_env, monkeypatch) -> None:
    """断言有效性证明（关键用例）。

    把「代码级拒答」与「URL 白名单」两道同时关掉，模型输出被直接放行 →
    不该出现的资源被推荐出来。此时 `expect_refusal` **必须**判失败。

    修复前这条断言在评测器里根本没实现：mock 返回的资源 URL 全部取自真实
    资料库，`resource_url_must_be_local` 查不出问题，于是关掉守卫后对抗集
    依然「全过」—— 一个空转的断言会把真实回归合法化。
    """
    monkeypatch.setenv("ABLATE_REFUSAL", "1")
    monkeypatch.setenv("ABLATE_WHITELIST", "1")
    monkeypatch.delenv("ABLATE_THRESHOLD", raising=False)

    _, cases = eval_suite.load_cases(ADVERSARIAL_PATH)
    local_urls = eval_suite.load_local_urls(CORPUS_PATH)
    report = eval_suite.run_cases(_graph_invoker(), cases, local_urls)

    s = report["summary"]
    assert s["passed"] == 0, "关掉守卫后应当全部判失败，否则断言是空转的"
    assert s["resource_count"] > 0, "模型输出应当被放行（这才构成『不该推荐却推荐了』）"
    assert all(any("期望拒答" in f for f in r["fails"]) for r in report["results"])

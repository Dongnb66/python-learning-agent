"""防幻觉评测集（L6）—— 把「不编造」从主观印象变成可量化、可进 CI 的断言。

为什么需要它
------------
「我的 RAG 能防幻觉」是一句无法验证的话。面试官无法证伪，也就无法相信。
本模块把口头承诺拆成 4 类**可判定的断言**，对整条管线做端到端契约测试：

1. **完整性**：5 类产物（profile / plan / resources / quiz / review）必须齐全；
2. **防幻觉命中率**：推荐资源的每一条 URL 都必须能在本地资料库里找到出处
   —— 出现一条编造链接即为违规；
3. **拒答正确性**：资料库确实没有相关话题时，resources 必须为空
   （空 = 正确的拒答，而不是编造几条凑数）；
4. **强拒答（expect_refusal）**：对抗用例可声明「本用例必须拒答」——
   `may_refuse_if_no_match` 只是「允许空」，空不空都算过；
   `expect_refusal` 要求**必须**为空。二者缺一不可：
   只有前者的话，拒答机制坏掉时用例照样通过（空转的断言）。
5. **画像准确度**：姓名 / 专业等确定性字段由正则层保底提取。

为什么需要 `expect_refusal`（一个真实的坑）
------------------------------------------
对抗集当初声明的就是这个语义，但评测器一直没有实现这个键 ——
于是即便三道约束全被关掉，对抗用例仍会「通过」：
mock 模型返回的资源全都取自真实资料库（URL 天然合法），
`resource_url_must_be_local` 也就查不出问题。**断言没生效 = 没有断言。**

另外还要注意触发条件：检索用的 query 来自**学习计划的步骤标题**，
而不是用户原话。因此在离线桩模式下，即使输入「量子计算」这类
资料库完全没有的话题，固定桩计划仍会让检索命中 —— 拒答分支根本走不到。
所以 `run_cases` 允许用例声明 `plan_titles`，由支持注入计划的调用器
（`GraphInvoker`）真正把话题送进检索，让这条断言有的放矢。

输出：控制台报告 + `eval/report_<时间戳>.json`，可直接挂到 CI，
把「能力」变成每次提交都会被验证的回归项。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Iterable

# 响应里必须齐全的产物字段
PRODUCT_FIELDS = ("profile", "plan", "resources", "quiz", "review")

DEFAULT_CASES_PATH = "eval/bad_cases.json"

# 调用器签名：(user_id, messages, case) -> 响应字典
# 第三个参数把整条用例交给调用器，使「用例声明计划」这类注入成为可能
# （见 GraphInvoker）；不需要它的实现忽略即可。
Invoker = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
def load_cases(path: str | pathlib.Path = DEFAULT_CASES_PATH) -> tuple[dict, list[dict]]:
    """读取评测集配置，返回 (全局配置, 用例列表)。"""
    cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return cfg, cfg["cases"]


def load_local_urls(path: str | pathlib.Path) -> set[str]:
    """读取本地资料库的全部 URL —— 防幻觉断言的「白名单真值」。

    兼容两种结构：`[{url}]` 或 `{"resources": [{url}]}`。
    """
    p = pathlib.Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("resources", data.get("items", []))
    return {it["url"] for it in items if isinstance(it, dict) and it.get("url")}


# --------------------------------------------------------------------------- #
# 单条用例判定
# --------------------------------------------------------------------------- #
def check_case(
    resp: dict[str, Any], case: dict, local_urls: set[str]
) -> tuple[bool, list[str], dict[str, Any]]:
    """返回 (是否通过, 失败原因, 指标)。"""
    m: dict[str, Any] = {}

    present = {f: resp.get(f) is not None for f in PRODUCT_FIELDS}
    m["products_present"] = present

    resources = resp.get("resources") or []
    urls = [r.get("url") for r in resources if isinstance(r, dict)]
    m["resource_count"] = len(urls)
    m["refused"] = len(urls) == 0

    # 防幻觉：每条 URL 都必须能在本地资料库找到出处
    fabricated = [u for u in urls if u and local_urls and u not in local_urls]
    m["fabricated_links"] = fabricated
    m["unverifiable_links"] = [u for u in urls if not u]

    a = case.get("assert", {})
    fails: list[str] = []

    if a.get("all_products_present") and not all(present.values()):
        fails.append(f"产物不齐: {present}")
    if a.get("resource_url_must_be_local"):
        if fabricated:
            fails.append(f"编造链接（不在资料库中）: {fabricated}")
        if m["unverifiable_links"]:
            fails.append(f"缺少出处 URL: {m['unverifiable_links']}")
    if "resource_count_min" in a and len(urls) < a["resource_count_min"]:
        fails.append(f"资源数 {len(urls)} < 期望 {a['resource_count_min']}")
    if "profile_name_equals" in a:
        got = (resp.get("profile") or {}).get("name")
        if got != a["profile_name_equals"]:
            fails.append(f"姓名期望 {a['profile_name_equals']!r} 实际 {got!r}")
    if "profile_major_equals" in a:
        got = (resp.get("profile") or {}).get("major")
        if got != a["profile_major_equals"]:
            fails.append(f"专业期望 {a['profile_major_equals']!r} 实际 {got!r}")

    # 强拒答：本用例「必须」拒答（区别于 may_refuse_if_no_match 的「允许拒答」）。
    # 少了这一条，三道约束被关掉时对抗用例仍会通过 —— 那就等于没测。
    if a.get("expect_refusal") and not m["refused"]:
        fails.append(f"期望拒答（资料库无相关内容），却返回了 {len(urls)} 条资源")

    # 允许拒答的用例：拒答本身不算失败（正确行为），记录下来用于统计拒答率
    m["refusal_expected"] = bool(a.get("may_refuse_if_no_match"))

    return (not fails), fails, m


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐条结果汇总成报告指标。"""
    n = len(results)
    passed = sum(1 for r in results if r.get("pass"))
    resources = sum(r.get("metrics", {}).get("resource_count", 0) for r in results)
    fabricated = sum(len(r.get("metrics", {}).get("fabricated_links", [])) for r in results)

    verifiable = resources - fabricated
    anti_hallucination = 100.0 * verifiable / resources if resources else 100.0

    refusal_eligible = [r for r in results if r.get("metrics", {}).get("refusal_expected")]
    refused = [r for r in refusal_eligible if r.get("metrics", {}).get("refused")]
    refusal_rate = 100.0 * len(refused) / len(refusal_eligible) if refusal_eligible else None

    return {
        "cases": n,
        "passed": passed,
        "completeness_pct": round(100.0 * passed / n, 1) if n else 0.0,
        "resource_count": resources,
        "fabricated_links": fabricated,
        "anti_hallucination_pct": round(anti_hallucination, 1),
        "refusal_eligible_cases": len(refusal_eligible),
        "refused_cases": len(refused),
        "refusal_rate_pct": None if refusal_rate is None else round(refusal_rate, 1),
    }


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def run_cases(
    invoker: Invoker,
    cases: Iterable[dict],
    local_urls: set[str],
    *,
    on_case: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """跑完全部用例，返回 {summary, results}。

    整条 case 会作为第三个参数传给调用器：需要注入学习计划（对抗用例要
    让话题真的进入检索）的实现可以对它做出反应，其余实现忽略即可。
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {"id": case.get("id", ""), "desc": case.get("desc", "")}
        try:
            resp = invoker(case.get("id", "eval-user"), case["messages"], case)
            ok, fails, metrics = check_case(resp, case, local_urls)
            entry["pass"] = ok
            entry["fails"] = fails
            entry["metrics"] = metrics
        except Exception as exc:  # 单条失败不中断整轮
            entry["pass"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["metrics"] = {}
        results.append(entry)
        if on_case:
            on_case(entry)

    return {"summary": summarize(results), "results": results}


def write_report(report: dict[str, Any], out_dir: str | pathlib.Path = "eval") -> pathlib.Path:
    """把报告落盘为 JSON（文件名带时间戳）。"""
    import datetime

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"report_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def default_invoker() -> Invoker:
    """基于 FastAPI TestClient 的进程内调用器（无需启动服务、无需网络）。

    依赖 httpx（FastAPI TestClient 所需），已在 requirements 中声明。

    注意：走 HTTP 的调用器**无法注入学习计划**，因此不适合跑声明了
    `plan_titles` 的对抗用例（固定桩计划会让检索照样命中，拒答分支走不到）。
    跑对抗集请用 `GraphInvoker`。
    """
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def _invoke(user_id: str, messages: list[dict[str, str]], case: dict[str, Any]) -> dict[str, Any]:
        r = client.post("/api/learn", json={"user_id": user_id, "messages": messages})
        r.raise_for_status()
        return r.json()

    return _invoke


class GraphInvoker:
    """直接驱动 LangGraph 的调用器：可在用例要求时注入学习计划。

    为什么需要注入计划
    ------------------
    检索用的 query 是**计划步骤的标题**，不是用户原话。离线桩模式下计划是
    固定桩（永远关于 Python / FastAPI），于是「用户问量子计算」这种对抗输入
    永远不会进入检索 —— 拒答分支形同不存在，改坏了也没人发现。

    用例只要声明 `plan_titles`，本调用器就用一个只产出该计划的 planner 节点
    重建图，让话题真正走到检索层。注入的计划来自用例数据（可见、可审），
    不是藏在校验逻辑里的暗门。

    每次调用同时留一条 trace（耗时 / 最慢节点 / 计数器），供复现报告统计。
    """

    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []

    @staticmethod
    def _state(user_id: str, messages: list[dict[str, str]], plan: Any = None) -> dict[str, Any]:
        from app.models import Message

        state: dict[str, Any] = {
            "user_id": user_id,
            "messages": [Message(role=m["role"], content=m["content"]) for m in messages],
            "resources": [],
            "errors": [],
        }
        if plan is not None:
            state["plan"] = plan
        return state

    @staticmethod
    def _stub_planner(titles: list[str], goal: str) -> Callable[[Any], dict[str, Any]]:
        """构造一个「只产出指定计划」的 planner 节点（替代 LLM 规划）。"""
        from app.models import Plan, PlanStep

        plan = Plan(
            goal=goal,
            steps=[
                PlanStep(step=i, title=t, description="用例注入的计划步骤", est_minutes=60)
                for i, t in enumerate(titles, start=1)
            ],
            total_minutes=60 * len(titles),
        )

        def _node(_state: Any) -> dict[str, Any]:
            return {"plan": plan}

        return _node

    @staticmethod
    def _dump(obj: Any) -> Any:
        return obj.model_dump() if hasattr(obj, "model_dump") else obj

    def __call__(self, user_id: str, messages: list[dict[str, str]], case: dict[str, Any]) -> dict[str, Any]:
        from app import telemetry
        from app.graph import build_graph

        titles = case.get("plan_titles") or []
        if titles:
            graph = build_graph(
                overrides={"planner": self._stub_planner(titles, case.get("plan_goal", "用例注入的计划"))}
            )
        else:
            graph = build_graph()

        with telemetry.trace_run() as tr:
            final = graph.invoke(self._state(user_id, messages))
        self.traces.append(
            {
                "case_id": case.get("id", ""),
                "duration_ms": tr.duration_ms,
                "slowest": tr.slowest(3),
                "counters": dict(tr.counters),
            }
        )
        return {
            "user_id": user_id,
            "profile": self._dump(final.get("profile")),
            "plan": self._dump(final.get("plan")),
            "resources": [self._dump(r) for r in (final.get("resources") or [])],
            "quiz": self._dump(final.get("quiz")),
            "review": self._dump(final.get("review")),
            "tutoring": self._dump(final.get("tutoring")),
            "errors": final.get("errors", []),
        }

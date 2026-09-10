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
4. **画像准确度**：姓名 / 专业等确定性字段由正则层保底提取。

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

Invoker = Callable[[str, list[dict[str, str]]], dict[str, Any]]


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
    """跑完全部用例，返回 {summary, results}。"""
    results: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {"id": case.get("id", ""), "desc": case.get("desc", "")}
        try:
            resp = invoker(case.get("id", "eval-user"), case["messages"])
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
    """
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    def _invoke(user_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        r = client.post("/api/learn", json={"user_id": user_id, "messages": messages})
        r.raise_for_status()
        return r.json()

    return _invoke

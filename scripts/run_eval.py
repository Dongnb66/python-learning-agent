#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""防幻觉评测 CLI —— 对整条管线跑一批用例，量化「不编造」这件事。

零配置用法（推荐，不需要 Key、不需要起服务）：

    python scripts/run_eval.py

它会自动：
  1. 若未配置可用的 LLM Key，则打开 `MOCK_LLM=1`（离线桩，不访问网络）；
  2. 用 FastAPI TestClient 在**进程内**跑完整管线（无需 uvicorn、无需端口）；
  3. 逐条打印判定结果，并把 JSON 报告写到 `eval/report_<时间戳>.json`。

对接真实模型（验证模型在约束下是否真的不编造）：

    # 在 .env 配好 LLM_API_KEY 后
    python scripts/run_eval.py

对已启动的服务做端到端评测：

    uvicorn app.main:app --port 8000
    python scripts/run_eval.py --base http://localhost:8000

退出码：全部通过为 0，否则为 1（便于挂 CI）。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# 让脚本在「仓库根目录」或任意 cwd 下都能 import app.*
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _usable_key(api_key: str | None) -> bool:
    """与 TutorAgent 同一套占位符判断，避免两处规则不一致。"""
    from app.agents.tutor_agent import _has_usable_key

    return _has_usable_key(api_key)


def _bootstrap_mock() -> bool:
    """无可用 Key 时打开 Mock 模式。必须在导入 app.main 之前设置。"""
    from app.config import get_settings

    if os.getenv("MOCK_LLM") is not None:
        return (os.getenv("MOCK_LLM") or "").strip().lower() in {"1", "true", "yes", "on"}
    if _usable_key(get_settings().llm_api_key):
        return False
    os.environ["MOCK_LLM"] = "1"
    return True


def _quiet_http_logs() -> None:
    """静音 HTTP 客户端日志。

    `app.main` 在导入时执行了 `logging.basicConfig(level=INFO)`，会把每次请求
    的 INFO 日志刷到控制台，把评测结果淹没。这里只调低 httpx 家族的级别，
    不影响项目自身的日志配置。
    """
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("httpx"):
            logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="python-learning-agent 防幻觉评测")
    ap.add_argument("--cases", default=str(_ROOT / "eval" / "bad_cases.json"))
    ap.add_argument("--base", default=None,
                    help="对已运行的服务做评测（如 http://localhost:8000）；省略则进程内运行")
    ap.add_argument("--resource-file", default=None, help="本地资料库路径（默认取用例文件里的配置）")
    ap.add_argument("-v", "--verbose", action="store_true", help="保留 HTTP 请求日志")
    args = ap.parse_args()

    from app import eval_suite

    mock_mode = False if args.base else _bootstrap_mock()

    base_url = args.base
    if base_url:
        # 走 HTTP：需要一个独立运行的服务，此处不改动本进程的 mock 状态
        import requests
        from urllib.parse import urljoin

        def invoker(user_id: str, messages: list[dict[str, str]]) -> dict:
            r = requests.post(
                urljoin(base_url.rstrip("/") + "/", "api/learn"),
                json={"user_id": user_id, "messages": messages},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()
    else:
        invoker = eval_suite.default_invoker()
        if not args.verbose:
            _quiet_http_logs()

    cfg, cases = eval_suite.load_cases(args.cases)
    res_path = args.resource_file or os.path.join(str(_ROOT), cfg.get("local_resource_file", "app/data/resources.json"))
    local_urls = eval_suite.load_local_urls(res_path)

    mode_label = "真实模型" if not mock_mode else "Mock（离线桩）"
    print(f"[评测] 模式={mode_label}  用例={len(cases)}  资料库白名单={len(local_urls)} 条 URL")
    if not local_urls:
        print(f"[警告] 未读到本地资料库（{res_path}），防幻觉断言将无法判定")
    print()

    def _on_case(entry: dict) -> None:
        flag = "PASS" if entry.get("pass") else "FAIL"
        print(f"  [{flag}] {entry['id']:26s} {entry.get('desc', '')}")
        if entry.get("error"):
            print(f"         ! {entry['error']}")
        for f in entry.get("fails", []) or []:
            print(f"         - {f}")
        metrics = entry.get("metrics") or {}
        if metrics:
            print(f"         资源 {metrics.get('resource_count', 0)} 条 | "
                  f"拒答={metrics.get('refused')} | 编造链接={len(metrics.get('fabricated_links', []))}")

    report = eval_suite.run_cases(invoker, cases, local_urls, on_case=_on_case)
    s = report["summary"]

    print("\n========== 评测报告 ==========")
    print(f"用例数            : {s['cases']}")
    print(f"通过数            : {s['passed']}  ({s['completeness_pct']}%)")
    print(f"推荐资源总数      : {s['resource_count']}")
    print(f"编造链接数        : {s['fabricated_links']}")
    print(f"防幻觉命中率       : {s['anti_hallucination_pct']}%  (来源可验证的资源占比)")
    if mock_mode:
        print("拒答正确率         : 见 tests/test_resource_guard.py")
        print("                     （mock 模式下计划为固定桩，检索总能命中，")
        print("                       无法触达拒答分支；已用确定性单测覆盖）")
    else:
        rate = s.get("refusal_rate_pct")
        print(f"拒答正确率         : {'n/a' if rate is None else f'{rate}%'}  "
              f"({s['refused_cases']}/{s['refusal_eligible_cases']} 条无匹配用例正确拒答)")
    print("==============================\n")

    path = eval_suite.write_report(report, out_dir=_ROOT / "eval")
    print(f"报告已写入: {path}")

    return 0 if s["passed"] == s["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())

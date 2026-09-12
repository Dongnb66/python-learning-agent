#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键复现 —— 把「这套指标是真的、而且你能自己跑出来」变成一条命令。

为什么需要它
------------
README 里的「单测全过 / 评测用例全过 / 编造链接 0」是**结论**。
结论本身没法自证：环境不同、依赖版本不同、改了代码没重跑，数字就可能是旧的。
本脚本做三件事，让结论可被独立复核：

1. **记账环境指纹**：Python 版本、平台、依赖版本、git commit、工作区是否干净、
   是否 Mock 模式 —— 报告离了这些上下文就只是一串没有意义的数字；
2. **一次跑完全部验证**：单测 → 常规评测 → 对抗评测 → 防幻觉消融，
   而不是分散在四处让人自己拼；
3. **失败即非零退出**：任何一环不达标都不允许「看起来跑完了」，
   可直接挂进 CI 当作质量闸门。

用法
----
    python scripts/reproduce.py                # 全跑
    python scripts/reproduce.py --skip-pytest  # 跳过单测（快速看评测）
    python scripts/reproduce.py --out eval/xx.json

输出：控制台汇总表 + `eval/reproduce_report.json`（默认覆盖，便于 diff 前后差异）
退出码：0 = 全部达标；1 = 有不达标项
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_OUT = _ROOT / "eval" / "reproduce_report.json"
NORMAL_CASES = _ROOT / "eval" / "bad_cases.json"
ADVERSARIAL_CASES = _ROOT / "eval" / "adversarial_cases.json"
CORPUS = _ROOT / "app" / "data" / "resources.json"

# 需要记录版本的包（缺了不算失败，照实记为 unknown）
_PKGS = ("fastapi", "starlette", "pydantic", "langchain", "langchain-core",
         "langgraph", "langchain-openai", "langchain-community", "rank-bm25",
         "sqlalchemy", "httpx", "pytest")


# --------------------------------------------------------------------------- #
# 环境指纹
# --------------------------------------------------------------------------- #
def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(_ROOT), capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except Exception:  # noqa: BLE001
        return "unavailable"


def environment_fingerprint() -> dict[str, Any]:
    """没有指纹的指标等于没有上下文 —— 复现不了就谈不上「可复现」。"""
    from importlib.metadata import PackageNotFoundError, version

    pkgs: dict[str, str] = {}
    for name in _PKGS:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = "not-installed"
        except Exception:  # noqa: BLE001
            pkgs[name] = "unknown"

    dirty = _git(["status", "--porcelain"])
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _git(["rev-parse", "HEAD"]) or "unavailable",
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unavailable",
        "git_dirty_files": 0 if dirty in ("", "unavailable") else len(dirty.splitlines()),
        "packages": pkgs,
    }


# --------------------------------------------------------------------------- #
# 第 1 环：单测
# --------------------------------------------------------------------------- #
def run_pytest() -> dict[str, Any]:
    basetemp = pathlib.Path(tempfile.mkdtemp(prefix="pla_reproduce_"))
    cmd = [
        sys.executable, "-m", "pytest", "-q", "--no-header",
        "-p", "no:cacheprovider", f"--basetemp={basetemp}",
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")

    passed = re.search(r"(\d+) passed", out)
    failed = re.search(r"(\d+) failed", out)
    errors = re.search(r"(\d+) error", out)
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "errors": int(errors.group(1)) if errors else 0,
        "tail": out.strip().splitlines()[-1][:200] if out.strip() else "",
    }


# --------------------------------------------------------------------------- #
# 第 2 环：评测集（常规 / 对抗）
# --------------------------------------------------------------------------- #
def run_eval_suite(cases_path: pathlib.Path) -> dict[str, Any]:
    """离线跑一张评测集，返回汇总 + 逐条结果 + 耗时统计。"""
    from sqlalchemy import create_engine

    from app import db, eval_suite

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pla_eval_"))
    engine = create_engine(f"sqlite:///{tmp / 'reproduce.db'}", future=True)
    db.get_engine = lambda: engine  # type: ignore[assignment]
    db.init_db()

    _, cases = eval_suite.load_cases(str(cases_path))
    local_urls = eval_suite.load_local_urls(str(CORPUS))
    # 用 GraphInvoker 而非 HTTP 调用器：对抗用例声明了 plan_titles，
    # 需要把话题真正注入计划，否则拒答分支永远走不到。
    invoker = eval_suite.GraphInvoker()
    report = eval_suite.run_cases(invoker, cases, local_urls)

    durations = [t["duration_ms"] for t in invoker.traces]
    slowest = invoker.traces[0]["slowest"] if invoker.traces else []
    summary = report["summary"]
    summary["avg_duration_ms"] = round(sum(durations) / len(durations), 1) if durations else 0.0
    summary["max_duration_ms"] = max(durations) if durations else 0.0
    summary["slowest_nodes_first_case"] = slowest
    return summary


# --------------------------------------------------------------------------- #
# 第 3 环：消融实验
# --------------------------------------------------------------------------- #
def run_ablation() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_ablation.py")],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    rows: list[dict[str, Any]] = []
    latest = sorted((_ROOT / "eval").glob("ablation_*.json"))
    if latest:
        try:
            rows = json.loads(latest[-1].read_text(encoding="utf-8")).get("rows", [])
        except Exception:  # noqa: BLE001
            rows = []
    baseline = rows[0] if rows else {}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "baseline_fabricated": baseline.get("编造链接_总数"),
        "groups": [
            {"group": r.get("group"), "fabricated": r.get("编造链接_总数"),
             "refused": r.get("脏输入_是否拒答")}
            for r in rows
        ],
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def _verdict(report: dict[str, Any]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    t = report.get("tests")
    if t is not None:
        if not t["ok"]:
            bad.append(f"单测未通过（{t['failed']} failed / {t['errors']} error）")
    for key, label in (("eval_normal", "常规评测"), ("eval_adversarial", "对抗评测")):
        s = report.get(key)
        if s is None:
            continue
        if s["passed"] != s["cases"]:
            bad.append(f"{label}未全过（{s['passed']}/{s['cases']}）")
        if s["fabricated_links"]:
            bad.append(f"{label}出现编造链接 {s['fabricated_links']} 条")
    ab = report.get("ablation")
    if ab is not None and not ab["ok"]:
        bad.append(f"消融实验基线组出现编造链接（exit={ab['exit_code']}）")
    return (not bad), bad


def _print_table(report: dict[str, Any]) -> None:
    fp = report["fingerprint"]
    print("=" * 78)
    print("python-learning-agent 一键复现")
    print("=" * 78)
    print(f"时间       : {fp['generated_at']}")
    print(f"Python     : {fp['python']} ({fp['python_impl']}) on {fp['platform']}")
    print(f"git        : {fp['git_branch']}@{fp['git_commit'][:12]}"
          f"  工作区未提交文件 {fp['git_dirty_files']} 个")
    print(f"Mock 模式  : {report['mock_llm']}")
    print("-" * 78)
    print(f"{'环节':<16}{'结果':<28}{'关键指标'}")
    print("-" * 78)

    t = report.get("tests")
    if t is not None:
        print(f"{'① 单元测试':<16}{'PASS' if t['ok'] else 'FAIL':<28}"
              f"{t['passed']} passed / {t['failed']} failed")

    for key, label in (("eval_normal", "② 常规评测"), ("eval_adversarial", "③ 对抗评测")):
        s = report.get(key)
        if s is None:
            continue
        res = "PASS" if s["passed"] == s["cases"] else "FAIL"
        print(f"{label:<16}{res:<28}"
              f"{s['passed']}/{s['cases']} 用例 · 编造链接 {s['fabricated_links']} · "
              f"防幻觉率 {s['anti_hallucination_pct']}% · 均值 {s['avg_duration_ms']}ms")

    ab = report.get("ablation")
    if ab is not None:
        print(f"{'④ 防幻觉消融':<16}{'PASS' if ab['ok'] else 'FAIL':<28}"
              f"全约束组编造链接 {ab['baseline_fabricated']} · 共 {len(ab['groups'])} 组对照")

    print("-" * 78)
    ok, bad = _verdict(report)
    if ok:
        print("结论：全部达标 ✅")
    else:
        print("结论：存在不达标项 ❌")
        for b in bad:
            print(f"  · {b}")


def main() -> int:
    ap = argparse.ArgumentParser(description="python-learning-agent 一键复现")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="报告输出路径")
    ap.add_argument("--skip-pytest", action="store_true", help="跳过单元测试（只看评测与消融）")
    ap.add_argument("--skip-ablation", action="store_true", help="跳过消融实验")
    args = ap.parse_args()

    # 全程离线：不依赖 API Key，也不访问网络
    os.environ["MOCK_LLM"] = "1"
    os.environ["TUTOR_MODE"] = "policy"
    os.environ.pop("ABLATE_THRESHOLD", None)
    os.environ.pop("ABLATE_REFUSAL", None)
    os.environ.pop("ABLATE_WHITELIST", None)

    report: dict[str, Any] = {"fingerprint": environment_fingerprint()}
    from app import telemetry

    report["mock_llm"] = telemetry.mock_mode()

    if not args.skip_pytest:
        print("→ 运行单元测试 ...", flush=True)
        report["tests"] = run_pytest()

    print("→ 运行常规评测（6 例）...", flush=True)
    report["eval_normal"] = run_eval_suite(NORMAL_CASES)

    if ADVERSARIAL_CASES.exists():
        print("→ 运行对抗评测 ...", flush=True)
        report["eval_adversarial"] = run_eval_suite(ADVERSARIAL_CASES)

    if not args.skip_ablation:
        print("→ 运行防幻觉消融实验 ...", flush=True)
        report["ablation"] = run_ablation()

    _print_table(report)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {out}")

    ok, _ = _verdict(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

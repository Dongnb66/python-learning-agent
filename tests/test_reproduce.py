"""一键复现脚本自身的测试。

「可复现」这件事本身也要可验证，否则它就是一句更响的声明：
- 环境指纹字段齐全吗（少了 git commit / 依赖版本，报告就失去了可对照性）？
- 判定逻辑真的会因不达标而失败吗（一个永远返回「达标」的闸门毫无意义）？
- 报告里的用例文件、资料库路径是否真实存在？

只测纯逻辑与常量，不真的去跑整套（那会拖慢每次 `pytest`）。
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_reproduce():
    spec = importlib.util.spec_from_file_location(
        "reproduce_script", _ROOT / "scripts" / "reproduce.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reproduce = _load_reproduce()


# --------------------------------------------------------------------------- #
# 环境指纹
# --------------------------------------------------------------------------- #
def test_fingerprint_has_all_context_fields() -> None:
    fp = reproduce.environment_fingerprint()
    for key in (
        "generated_at",
        "python",
        "python_impl",
        "platform",
        "git_commit",
        "git_branch",
        "git_dirty_files",
        "packages",
    ):
        assert key in fp, f"指纹缺少 {key}，报告将无法对照"

    import sys

    assert fp["python"] == sys.version.split()[0]
    assert isinstance(fp["git_dirty_files"], int)


def test_fingerprint_records_dependency_versions() -> None:
    """依赖版本是复现的前提：同一份代码在不同 langgraph 上结果可能不同。"""
    pkgs = reproduce.environment_fingerprint()["packages"]
    assert "fastapi" in pkgs and "langgraph" in pkgs
    # 没装的包如实记 not-installed，而不是静默省略
    assert all(isinstance(v, str) and v for v in pkgs.values())


# --------------------------------------------------------------------------- #
# 判定闸门
# --------------------------------------------------------------------------- #
def _base_report() -> dict:
    return {
        "tests": {"ok": True, "exit_code": 0, "passed": 127, "failed": 0, "errors": 0, "tail": ""},
        "eval_normal": {"passed": 6, "cases": 6, "fabricated_links": 0},
        "eval_adversarial": {"passed": 3, "cases": 3, "fabricated_links": 0},
        "ablation": {"ok": True, "exit_code": 0},
    }


def test_verdict_passes_on_clean_report() -> None:
    ok, bad = reproduce._verdict(_base_report())
    assert ok and bad == []


def test_verdict_fails_when_tests_fail() -> None:
    r = _base_report()
    r["tests"].update(ok=False, exit_code=1, failed=2)
    ok, bad = reproduce._verdict(r)
    assert not ok and any("单测" in b for b in bad)


@pytest.mark.parametrize("key", ["eval_normal", "eval_adversarial"])
def test_verdict_fails_on_incomplete_or_fabricated_links(key: str) -> None:
    r = _base_report()
    r[key].update(passed=2, fabricated_links=1)
    ok, bad = reproduce._verdict(r)
    assert not ok
    assert any("编造链接" in b for b in bad)
    assert any("未全过" in b for b in bad)


def test_verdict_fails_when_ablation_baseline_leaks() -> None:
    r = _base_report()
    r["ablation"].update(ok=False, exit_code=1)
    ok, bad = reproduce._verdict(r)
    assert not ok and any("消融" in b for b in bad)


def test_verdict_ignores_stages_that_were_skipped() -> None:
    """--skip-pytest 时报告里没有 tests 键，不能因此误判为失败。"""
    r = _base_report()
    r.pop("tests")
    ok, _ = reproduce._verdict(r)
    assert ok


# --------------------------------------------------------------------------- #
# 路径常量
# --------------------------------------------------------------------------- #
def test_reproduce_targets_existing_artifacts() -> None:
    assert reproduce.NORMAL_CASES.exists()
    assert reproduce.CORPUS.exists()
    assert reproduce.DEFAULT_OUT.parent == _ROOT / "eval"


def test_adversarial_cases_are_present_and_counted() -> None:
    """对抗集在库但默认运行器不跑 —— 复现脚本要把它跑起来，否则等于没有。"""
    import json

    assert reproduce.ADVERSARIAL_CASES.exists()
    data = json.loads(reproduce.ADVERSARIAL_CASES.read_text(encoding="utf-8"))
    assert len(data["cases"]) >= 1
    assert all("assert" in c for c in data["cases"])

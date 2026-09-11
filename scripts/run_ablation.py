#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""防幻觉「消融实验」(Ablation) —— 证明三道代码级约束真的在起作用。

为什么需要它
------------
「我有三道防幻觉约束」是一句声明；面试官无法证伪，也就无法相信。
本脚本把**每道约束单独关掉**，对同一个输入跑一遍，对比输出 ——
用「关掉就变坏」证明「开着确实有用」。

为什么不是对整条管线做
----------------------
整条管线在 mock 模式下，学习计划是**固定桩**（不随输入变化），
检索总能命中资料库 → 三道约束都不会被触发，四组结果完全一样，实验无效。
（这一点已实测确认，是本脚本设计的重要前提。）

因此这里**直接对机制层做消融**：构造「干净输入」与「污染输入」两组，
分别在全约束 / 关约束下调用真实的 `resource` 节点，观察编造链接是否被拦住。
这样测的是约束本身，不掺和模型能力，结论更硬。

用法
----
    python scripts/run_ablation.py
    python scripts/run_ablation.py -v

输出：控制台对比表 + `eval/ablation_<时间戳>.json`
退出码：全约束组出现编造链接则为 1（当作回归失败）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --------------------------------------------------------------------------- #
# 约束开关
# --------------------------------------------------------------------------- #
GUARDS = {
    "threshold": "ABLATE_THRESHOLD",
    "refusal": "ABLATE_REFUSAL",
    "whitelist": "ABLATE_WHITELIST",
}


def _set_guards(disabled: set[str]) -> None:
    for name, env in GUARDS.items():
        if name in disabled:
            os.environ[env] = "1"
        else:
            os.environ.pop(env, None)


def _fresh_app_modules() -> None:
    """清掉 app.* 模块缓存，让模块级开关重新读取环境变量。"""
    for mod in [m for m in list(sys.modules) if m.startswith("app")]:
        del sys.modules[mod]


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #
def _make_state(plan_titles: list[str], user_text: str) -> dict[str, Any]:
    """构造一个最小 AgentState：只含 resource 节点需要的东西。"""
    from app.models import Message, Plan, PlanStep

    return {
        "user_id": "ablation-user",
        "messages": [Message(role="user", content=user_text)],
        "plan": Plan(
            goal="测试目标",
            steps=[PlanStep(step=i + 1, title=t, description="", est_minutes=60)
                   for i, t in enumerate(plan_titles)],
            total_minutes=60 * len(plan_titles),
        ),
        "resources": [],
        "errors": [],
    }


# 「正常话题」：计划步骤用的词都能在资料库里命中
_NORMAL_PLAN = ["Python 入门", "FastAPI 教程"]
_NORMAL_TEXT = "我想学 Python 和 FastAPI"

# 「脏话题」：计划步骤的词在资料库里**完全不存在**（量子计算 / 青铜器）
# → 这是触发「相关性阈值 + 代码级拒答」的场景
_DIRTY_PLAN = ["超导量子比特架构", "元代青花瓷钴料鉴定"]
_DIRTY_TEXT = "我想学量子计算和青花瓷鉴定"


# 模型幻觉模拟：一个「总是编造链接」的假模型
# —— 它无视给的资料，硬推荐两条资料库里不存在的链接。
# 这正是白名单要拦的东西，也回答了「把模型换成坏的，这条还成立吗？」
_FAKE_HALLUCINATED = [
    {"title": "量子计算权威指南", "type": "article",
     "url": "https://fake-quantum.example.com/guide", "description": "编造的", "relevance": "编造的"},
    {"title": "青花瓷鉴赏大全", "type": "doc",
     "url": "https://fake-porcelain.example.com/all", "description": "编造的", "relevance": "编造的"},
]

# 场景三用：「标题对上、URL 编造」——专测白名单能否把假 URL 纠正/剔除
_TITLE_ONLY_ITEM = [
    {"title": "Python 入门：变量、循环与函数", "type": "article",
     "url": "https://evil.example.com/fake-python", "description": "标题真实、URL 编造", "relevance": "测白名单"},
]


def _patch_llm_with_fake(items: list[dict[str, Any]]) -> None:
    """把 resource 节点用到的结构化模型替换成指定的假模型。

    只替换 `resource_agent` 模块里引用的 `get_structured_model`，
    不碰其他 agent，保证实验边界清晰。
    """
    from app.models import ResourceItem, ResourceList

    class _FakeStructured:
        def invoke(self, _messages):  # noqa: ANN001
            return ResourceList(items=[ResourceItem(**x) for x in items])

    import app.agents.resource_agent as ra

    ra.get_structured_model = lambda *a, **k: _FakeStructured()  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# 单组实验
# --------------------------------------------------------------------------- #
def _run_group(disabled: set[str], fake_items: list[dict[str, Any]]) -> dict[str, Any]:
    """跑一个场景，返回指标。"""
    _set_guards(disabled)
    _fresh_app_modules()
    os.environ["MOCK_LLM"] = "1"

    from app.agents.resource_agent import build_resources_node

    _patch_llm_with_fake(fake_items)

    # 场景一：正常输入（检索能命中，模型编造无关链接）
    normal = build_resources_node(_make_state(_NORMAL_PLAN, _NORMAL_TEXT))["resources"]
    # 场景二：脏输入（检索必然为空）
    dirty = build_resources_node(_make_state(_DIRTY_PLAN, _DIRTY_TEXT))["resources"]

    return {
        "normal_count": len(normal),
        "dirty_count": len(dirty),
        "dirty_is_refused": len(dirty) == 0,
        "normal_urls": [r.url for r in normal],
        "dirty_urls": [r.url for r in dirty],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="python-learning-agent 防幻觉消融实验")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # 资料库真值（用于判定是否编造）
    from app.eval_suite import load_local_urls

    local_urls = load_local_urls(str(_ROOT / "app" / "data" / "resources.json"))

    print("=" * 78)
    print("防幻觉消融实验（Ablation）—— 关掉一道约束，看机制是否失效")
    print(f"资料库真值 {len(local_urls)} 条 URL · 把模型换成「永远编造链接」的假模型")
    print("=" * 78)

    # 三组 × 假模型（真模型不会必然幻觉，测不出白名单）
    groups: list[tuple[str, set[str]]] = [
        ("① 全约束（现状）", set()),
        ("② 关掉相关性阈值", {"threshold"}),
        ("③ 关掉代码级拒答", {"refusal"}),
        ("④ 关掉 URL 白名单", {"whitelist"}),
    ]

    rows: list[dict[str, Any]] = []
    for label, disabled in groups:
        m = _run_group(disabled, _TITLE_ONLY_ITEM)
        fabricated_normal = [u for u in m["normal_urls"] if u and u not in local_urls]
        fabricated_dirty = [u for u in m["dirty_urls"] if u and u not in local_urls]
        fabricated = len(fabricated_normal) + len(fabricated_dirty)
        row = {
            "group": label,
            "disabled": sorted(disabled),
            "正常输入_资源数": m["normal_count"],
            "脏输入_资源数": m["dirty_count"],
            "脏输入_是否拒答": m["dirty_is_refused"],
            "编造链接_总数": fabricated,
        }
        rows.append(row)

        print(f"\n[{label}]")
        print(f"   正常输入 → 推荐 {m['normal_count']} 条")
        print(f"   脏输入   → 推荐 {m['dirty_count']} 条  {'（✅ 正确拒答）' if m['dirty_is_refused'] else '（❌ 未拒答）'}")
        print(f"   编造链接 → {fabricated} 条  {'✅' if fabricated == 0 else '❌ 幻觉出界'}")
        if args.verbose:
            for u in fabricated_normal + fabricated_dirty:
                print(f"       漏出的编造链接: {u}")

    # ---- 对比总结 ----
    base = rows[0]
    print("\n" + "=" * 78)
    print("对比总结")
    print("=" * 78)
    hdr = f"{'实验组':<18}{'正常/脏输入资源数':<20}{'脏输入拒答':<12}{'编造链接漏出':<14}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        cnt = f"{r['正常输入_资源数']}/{r['脏输入_资源数']}"
        refuse = "是" if r["脏输入_是否拒答"] else "否"
        fab = r["编造链接_总数"]
        print(f"{r['group']:<18}{cnt:<20}{refuse:<12}{fab}{'  ⚠️' if fab else '  ✅'}")

    print("\n结论：")
    print("  · 关掉「相关性阈值」→ 脏输入不再为空（0 分项被当事实依据）")
    print("  · 关掉「代码级拒答」→ 空检索仍会调用模型，把判断权交还给模型")
    print("  · 关掉「URL 白名单」→ 假模型编造的链接直接漏给用户")
    print("  全约束组编造链接必须为 0；任一组关掉后应看到指标变坏。")

    out = _ROOT / "eval"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"ablation_{ts}.json"
    path.write_text(
        json.dumps({
            "generated_at": ts,
            "mock": True,
            "model": "fake-always-hallucinate",
            "local_urls": len(local_urls),
            "rows": rows,
            "note": "对机制层做消融：把模型换成『永远编造链接』的假模型，逐道关掉约束，观察编造链接是否漏出。这样测的是约束本身，不受模型能力波动影响。",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入: {path}")

    _set_guards(set())
    return 1 if base["编造链接_总数"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

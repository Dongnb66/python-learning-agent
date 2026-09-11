"""防幻觉三道代码级约束的消融回归测试。

为什么需要这个测试
------------------
"我有三道防幻觉约束"是**声明**；只有证明"关掉它就会变坏"，才变成**机制**。
本测试对每条约束做一次消融（用环境变量关掉），断言：

1. 全约束开启时：脏输入（资料库无匹配）→ 必须拒答、编造链接必须为 0；
2. 关掉「URL 白名单」后：假模型编造的链接**确实会漏出**（证明白名单在拦东西）；
3. 关掉「相关性阈值」后：脏输入不再返回空（证明阈值在拦 0 分项）。

第 2、3 条是关键：它们证明约束**不是装饰**——关掉之后指标必然变坏。
所有测试都不依赖真实 LLM（用假模型），因此可在 CI 上确定性运行。
"""
from __future__ import annotations

import importlib
import os

import pytest

from app.models import Message, Plan, PlanStep, ResourceItem, ResourceList

# ---------------------------------------------------------------- 场景构造 --
# 脏输入：这些词在 app/data/resources.json 里完全不存在 → 检索必然为空
_DIRTY_PLAN = ["超导量子比特架构", "元代青花瓷钴料鉴定"]
_DIRTY_TEXT = "我想学量子计算和青花瓷鉴定"

# 正常输入：能命中资料库
_NORMAL_PLAN = ["Python 入门"]
_NORMAL_TEXT = "我想学 Python"

# 假模型：标题真实（能对上白名单）、URL 编造 —— 专测白名单
_FAKE_TITLE_OK_URL_FAKE = [
    {
        "title": "Python 入门：变量、循环与函数",
        "type": "article",
        "url": "https://evil.example.com/fake-python",
        "description": "标题真实、URL 编造",
        "relevance": "测白名单",
    }
]

# 假模型：完全是资料库里没有的东西
_FAKE_ALL_FAKE = [
    {
        "title": "量子计算权威指南",
        "type": "article",
        "url": "https://fake-quantum.example.com/guide",
        "description": "编造的",
        "relevance": "编造的",
    }
]


def _state(plan_titles: list[str], text: str) -> dict:
    return {
        "user_id": "ablation-test",
        "messages": [Message(role="user", content=text)],
        "plan": Plan(
            goal="测试",
            steps=[
                PlanStep(step=i + 1, title=t, description="", est_minutes=60)
                for i, t in enumerate(plan_titles)
            ],
            total_minutes=60 * len(plan_titles),
        ),
        "resources": [],
        "errors": [],
    }


@pytest.fixture
def guarded(monkeypatch):
    """返回一个函数：给定 fake items 和要关掉的约束集合，跑出资源列表。

    - 每组都重新 import `app.agents.resource_agent`，让模块级开关重新读取
      环境变量（否则第一次 import 后开关就被固化了）；
    - 只 monkeypatch 该模块引用的 `get_structured_model`，边界清晰。
    """
    os.environ["MOCK_LLM"] = "1"

    def _run(fake_items: list[dict], plan_titles: list[str], text: str,
             disabled: set[str] | None = None):
        disabled = disabled or set()
        env_map = {
            "threshold": "ABLATE_THRESHOLD",
            "refusal": "ABLATE_REFUSAL",
            "whitelist": "ABLATE_WHITELIST",
        }
        for name, env in env_map.items():
            if name in disabled:
                monkeypatch.setenv(env, "1")
            else:
                monkeypatch.delenv(env, raising=False)

        mod = importlib.reload(importlib.import_module("app.agents.resource_agent"))

        class _Fake:
            def invoke(self, _messages):
                return ResourceList(items=[ResourceItem(**x) for x in fake_items])

        monkeypatch.setattr(mod, "get_structured_model", lambda *a, **k: _Fake())
        return mod.build_resources_node(_state(plan_titles, text))["resources"]

    yield _run
    for env in ("ABLATE_THRESHOLD", "ABLATE_REFUSAL", "ABLATE_WHITELIST"):
        os.environ.pop(env, None)


# ---------------------------------------------------------------- 约束断言 --
def test_all_guards_on_dirty_input_refuses(guarded):
    """全约束开启：脏输入必须拒答（返回空），编造链接 0 条。"""
    res = guarded(_FAKE_ALL_FAKE, _DIRTY_PLAN, _DIRTY_TEXT)
    assert res == [], f"脏输入应拒答（返回空资源），实际返回 {len(res)} 条"


def test_all_guards_on_blocks_faked_url(guarded):
    """全约束开启：模型给的编造 URL 必须被白名单拦住（或纠正为真实 URL）。"""
    res = guarded(_FAKE_TITLE_OK_URL_FAKE, _NORMAL_PLAN, _NORMAL_TEXT)
    for r in res:
        assert r.url != "https://evil.example.com/fake-python", (
            "编造 URL 竟然漏给了用户 —— 白名单失效"
        )


def test_disable_whitelist_leaks_faked_url(guarded):
    """⭐ 关键断言：关掉白名单后，编造 URL 确实会漏出。

    这条证明「白名单不是装饰」——关掉它，防护就出现缺口。
    """
    res = guarded(_FAKE_TITLE_OK_URL_FAKE, _NORMAL_PLAN, _NORMAL_TEXT,
                  disabled={"whitelist"})
    urls = [r.url for r in res]
    assert "https://evil.example.com/fake-python" in urls, (
        "关掉白名单后编造 URL 仍未漏出 —— 说明该约束没有真正在起作用"
    )


def test_disable_threshold_makes_dirty_retrieval_nonempty(guarded):
    """⭐ 关键断言：关掉相关性阈值后，脏话题也能检索到东西（0 分项被放行）。

    阈值是拒答的**前置条件**：没有阈值，检索永远非空，`resource` 节点里的
    「context 为空 → 拒答」分支就永不触发。

    注意这里要在**检索层**验证，不能只看节点输出 —— 节点输出还受白名单影响，
    白名单会把不匹配的条目滤掉，从而掩盖阈值的作用。
    """
    import importlib as _il

    def _retrieve_count(with_ablate: bool) -> int:
        if with_ablate:
            os.environ["ABLATE_THRESHOLD"] = "1"
        else:
            os.environ.pop("ABLATE_THRESHOLD", None)
        mod = _il.reload(_il.import_module("app.rag"))
        return len(mod.retrieve("超导量子比特架构", k=4))

    try:
        with_guard = _retrieve_count(False)
        without_guard = _retrieve_count(True)
    finally:
        os.environ.pop("ABLATE_THRESHOLD", None)

    assert with_guard == 0, "全约束下，脏话题不应检索到任何资料"
    assert without_guard > 0, (
        "关掉阈值后脏话题仍检索不到东西 —— 说明阈值约束没有真正在起作用"
    )


def test_normal_input_still_works(guarded):
    """回归：正常输入下管线仍能产出资源（约束没有误伤正常路径）。"""
    res = guarded(_FAKE_TITLE_OK_URL_FAKE, _NORMAL_PLAN, _NORMAL_TEXT)
    assert len(res) >= 1
    # 只给标题、URL 编造 → 白名单应把它归一化成资料库里的真实 URL
    assert res[0].url != "https://evil.example.com/fake-python"

"""防幻觉三道硬约束的单测（确定性、零 Key、不访问网络）。

「我的 RAG 能防幻觉」若只靠 prompt 叮嘱模型，是无法验证的承诺。
本文件证明它其实是**代码层的不变量**：

1. 检索层：无相关命中时返回空列表，而不是拿 0 分项凑数；
2. 拒答层：检索为空 → 直接返回空资源，**根本不调用模型**；
3. 白名单层：模型即便返回编造链接，也会被逐条剔除；只有标题的会被
   归一化成资料库里的真实 URL。

三条合起来：管线**不可能**输出一条本地资料库里不存在的链接。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agents.resource_agent import build_resources_node
from app.models import Plan, PlanStep, ResourceItem, ResourceList
from app.rag import retrieve

# 资料库中真实存在的条目（取自 app/data/resources.json）
_REAL_TITLE = "LangGraph 多智能体编排入门"
_REAL_URL = "https://langchain-ai.github.io/langgraph/"
_FAKE_URL = "https://definitely-not-in-corpus.example.com/awesome-course"


def _plan(*titles: str) -> Plan:
    return Plan(
        goal="测试用计划",
        steps=[
            PlanStep(step=i, title=t, description="", est_minutes=60)
            for i, t in enumerate(titles, start=1)
        ],
        total_minutes=60 * len(titles),
    )


def _state(plan: Plan | None) -> dict:
    return {"user_id": "guard-user", "messages": [], "plan": plan, "errors": []}


# --------------------------------------------------------------------------- #
# 1. 检索层：相关性阈值
# --------------------------------------------------------------------------- #
def test_retrieve_returns_empty_for_topic_absent_from_corpus() -> None:
    """资料库里没有量子计算 → 必须返回空，而不是硬塞 4 条不相关结果。"""
    assert retrieve("量子计算基础") == []
    assert retrieve("我想学点编程") == []


def test_retrieve_keeps_relevant_hits_ranked_first() -> None:
    """有相关命中时必须保留，且最相关的一条排在最前。"""
    docs = retrieve("学习 LangGraph")
    assert docs, "LangGraph 是资料库里的真实条目，应能检索到"
    assert docs[0].metadata["title"] == _REAL_TITLE


# --------------------------------------------------------------------------- #
# 2. 拒答层：检索为空 → 不调用模型
# --------------------------------------------------------------------------- #
def test_empty_retrieval_refuses_and_never_calls_llm() -> None:
    """无事实依据就不生成——这是机制，不是期望。

    关键断言是「模型一次都没被调用」：既拒绝编造，也省掉一次无效请求。
    """
    fake_llm = MagicMock()
    with patch("app.agents.resource_agent.get_structured_model", fake_llm):
        out = build_resources_node(_state(_plan("量子纠缠的拓扑保护与容错")))
        fake_llm.assert_not_called()

    assert out["resources"] == []


def test_no_plan_also_refuses() -> None:
    """没有计划就没有检索词 → 同样拒答，不做无依据生成。"""
    fake_llm = MagicMock()
    with patch("app.agents.resource_agent.get_structured_model", fake_llm):
        out = build_resources_node(_state(None))
        fake_llm.assert_not_called()
    assert out["resources"] == []


# --------------------------------------------------------------------------- #
# 3. 白名单层：模型幻觉也拦得住
# --------------------------------------------------------------------------- #
def _llm_returning(items: list[ResourceItem]) -> MagicMock:
    m = MagicMock()
    m.return_value.invoke.return_value = ResourceList(items=items)
    return m


def test_whitelist_drops_fabricated_links() -> None:
    """模型返回一条真链接 + 一条编造链接 → 编造的必须被剔除。"""
    items = [
        ResourceItem(title=_REAL_TITLE, url=_REAL_URL, type="doc", description="真实资料"),
        ResourceItem(title="史上最强速成课", url=_FAKE_URL, type="video", description="幻觉产物"),
    ]
    with patch("app.agents.resource_agent.get_structured_model", _llm_returning(items)):
        out = build_resources_node(_state(_plan("学习 LangGraph")))

    urls = [r.url for r in out["resources"]]
    assert _FAKE_URL not in urls, "编造链接必须被白名单拦下"
    assert urls == [_REAL_URL]


def test_whitelist_normalizes_title_only_item_to_corpus_url() -> None:
    """模型只给了标题（偷懒 / 记不清 URL）→ 用资料库真实 URL 补上，而不是放行空链接。"""
    items = [ResourceItem(title=_REAL_TITLE, url=None, type="doc", description="有标题无链接")]
    with patch("app.agents.resource_agent.get_structured_model", _llm_returning(items)):
        out = build_resources_node(_state(_plan("学习 LangGraph")))

    assert [r.url for r in out["resources"]] == [_REAL_URL]


def test_whitelist_drops_unknown_title_without_url() -> None:
    """标题也不在资料库里 → 无从溯源，直接丢弃。"""
    items = [ResourceItem(title="某不存在的秘籍", url=None, type="doc", description="无出处")]
    with patch("app.agents.resource_agent.get_structured_model", _llm_returning(items)):
        out = build_resources_node(_state(_plan("学习 LangGraph")))

    assert out["resources"] == []


@pytest.mark.parametrize("bad_url", [_FAKE_URL, "http://localhost:9999/x", "javascript:alert(1)"])
def test_whitelist_is_url_exact_match(bad_url: str) -> None:
    """白名单是精确 URL 匹配：只有资料库里的 URL 能通过。"""
    items = [ResourceItem(title="伪装资料", url=bad_url, type="doc", description="")]
    with patch("app.agents.resource_agent.get_structured_model", _llm_returning(items)):
        out = build_resources_node(_state(_plan("学习 LangGraph")))
    assert out["resources"] == []

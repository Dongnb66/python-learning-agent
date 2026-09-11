"""ResourceAgent —— RAG 增强的个性化资源推荐（防幻觉核心）。

防幻觉不是「在 prompt 里叮嘱模型别编」，而是**代码层的三道硬约束**：

1. **检索层留真**：BM25 带相关性阈值（`min_score`），只保留真正命中的资料；
   资料库没有相关内容时就返回空，绝不拿 0 分项凑数。
2. **代码级拒答**：检索结果为空 → 直接返回空资源列表，**根本不调用模型**。
   没有事实依据就不生成——这是机制，不是期望。
3. **白名单兜底**：模型返回的每一条，其 URL 必须来自本轮检索到的候选集
   （或标题能对上资料库），否则一律丢弃；只给了标题的会被归一化成
   资料库里的真实 URL。**模型就算幻觉，也过不了这一关。**

三层都过了，返回的每一条链接都能在本地资料库里找到出处。
"""
from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Plan, ResourceItem, ResourceList
from app.rag import retrieve

# ---- 消融实验开关（仅供 scripts/run_ablation.py 做对照实验，生产不设置）----
def _refusal_disabled() -> bool:
    """`ABLATE_REFUSAL=1`：关掉「检索为空 → 代码级拒答」这道约束。"""
    return os.getenv("ABLATE_REFUSAL", "") == "1"


def _whitelist_disabled() -> bool:
    """`ABLATE_WHITELIST=1`：关掉 URL 白名单过滤这道约束。"""
    return os.getenv("ABLATE_WHITELIST", "") == "1"

_RESOURCE_SYSTEM = """你是「学习资源推荐智能体」。
下面给出了「检索到的真实资料库片段」和「学生的学习计划」。
请为计划中的关键步骤各推荐 1~2 条资料，总共返回 2~4 条 ResourceItem。
硬性要求：
- 只能从检索片段里挑选，禁止编造资料或链接；
- 每条写清 title、type(article/video/course/doc)、url、description、relevance（说明为何适配该学生）；
- 输出符合 ResourceItem 列表 schema 的 JSON。"""

_K = 4


def _collect_context(plan: Plan | None) -> tuple[str, dict[str, str]]:
    """按计划步骤逐一检索，返回 (给模型的资料全文, {标题: URL} 白名单)。"""
    queries = [step.title for step in (plan.steps if plan else [])]
    seen: set[str] = set()
    parts: list[str] = []
    allowed: dict[str, str] = {}  # title -> url，即本轮「可推荐白名单」
    for q in queries:
        for doc in retrieve(q, k=_K):
            title = doc.metadata.get("title", "")
            if title and title not in seen:
                seen.add(title)
                allowed[title] = doc.metadata.get("url", "")
                parts.append(
                    f"[{doc.metadata.get('type', '')}] {title} "
                    f"({doc.metadata.get('url', '')})\n{doc.page_content}"
                )
    return "\n\n".join(parts), allowed


def _enforce_whitelist(items: list[ResourceItem], allowed: dict[str, str]) -> list[ResourceItem]:
    """只放行有真实出处的条目；缺 URL 但标题能对上的，补成资料库真实 URL。"""
    allowed_urls = set(allowed.values())
    kept: list[ResourceItem] = []
    for it in items:
        if it.url and it.url in allowed_urls:
            kept.append(it)
        elif it.title and it.title in allowed:
            kept.append(it.model_copy(update={"url": allowed[it.title]}))
    return kept


def build_resources_node(state: AgentState) -> dict:
    plan: Plan = state.get("plan")

    # 第 1 步：只从本地资料库检索真实资料（带相关性阈值）
    context, allowed = _collect_context(plan)

    # 第 2 步：资料库确实没有相关内容 → 代码级拒答，不调用模型、不编造
    if not context and not _refusal_disabled():
        return {"resources": []}

    plan_text = plan.model_dump_json(ensure_ascii=False) if plan else "（暂无计划）"
    messages = (
        [SystemMessage(content=_RESOURCE_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [
            HumanMessage(
                content=f"【检索到的真实资料】\n{context or '（本轮未检索到资料）'}\n\n"
                f"【学习计划】\n{plan_text}\n\n请基于以上真实资料推荐资源 JSON 列表。"
            )
        ]
    )
    result: ResourceList = get_structured_model(ResourceList, temperature=0.2).invoke(messages)

    # 第 3 步：白名单过滤——模型幻觉出的链接在此被剔除
    if _whitelist_disabled():
        return {"resources": result.items}
    return {"resources": _enforce_whitelist(result.items, allowed)}

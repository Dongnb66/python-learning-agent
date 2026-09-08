"""ResourceAgent —— RAG 增强的个性化资源推荐（防幻觉核心）。

流程：
1. 从学习计划每一步提取检索词；
2. 用 BM25 在本地资料库检索真实存在的资料片段；
3. 把这些「事实依据」连同计划一起交给 LLM，让它只从检索结果中精选 2~3 条推荐。

由于 LLM 只能看到检索到的真实资料，无法凭空编造链接，从而抑制幻觉。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Plan, ResourceItem, ResourceList
from app.rag import retrieve

_RESOURCE_SYSTEM = """你是「学习资源推荐智能体」。
下面给出了「检索到的真实资料库片段」和「学生的学习计划」。
请为计划中的关键步骤各推荐 1~2 条资料，总共返回 2~4 条 ResourceItem。
硬性要求：
- 只能从检索片段里挑选，禁止编造资料或链接；
- 每条写清 title、type(article/video/course/doc)、url、description、relevance（说明为何适配该学生）；
- 输出符合 ResourceItem 列表 schema 的 JSON。"""

_K = 4


def build_resources_node(state: AgentState) -> dict:
    chat = get_structured_model(ResourceList, temperature=0.2)
    plan: Plan = state.get("plan")
    plan_text = plan.model_dump_json(ensure_ascii=False) if plan else "（暂无计划）"

    # 以计划标题/描述为检索词，拼接去重后的真实片段
    queries = [step.title for step in (plan.steps if plan else [])]
    seen: set[str] = set()
    ctx_parts: list[str] = []
    for q in queries:
        for doc in retrieve(q, k=_K):
            key = doc.metadata.get("title", "")
            if key and key not in seen:
                seen.add(key)
                ctx_parts.append(
                    f"[{doc.metadata.get('type','')}] {doc.metadata.get('title','')} "
                    f"({doc.metadata.get('url','')})\n{doc.page_content}"
                )
    context = "\n\n".join(ctx_parts) if ctx_parts else "（资料库为空）"

    messages = (
        [SystemMessage(content=_RESOURCE_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [
            HumanMessage(
                content=f"【检索到的真实资料】\n{context}\n\n"
                f"【学习计划】\n{plan_text}\n\n请基于以上真实资料推荐资源 JSON 列表。"
            )
        ]
    )
    result: ResourceList = chat.invoke(messages)
    return {"resources": result.items}

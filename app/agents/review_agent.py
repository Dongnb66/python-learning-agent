"""ReviewAgent —— 学情复盘与下一步建议。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Plan, Quiz, Review

_REVIEW_SYSTEM = """你是「学情复盘智能体」。
结合学生画像、学习计划与自测题，输出复盘报告：
- mastery：对学生当前掌握度的总体判断；
- strengths：已掌握的亮点（2~3 条）；
- gaps：仍存在的薄弱项（2~3 条）；
- suggestions：下一步学习建议（2~4 条）。
输出符合 Review schema 的 JSON。"""


def build_review_node(state: AgentState) -> dict:
    chat = get_structured_model(Review, temperature=0.3)
    profile_json = state["profile"].model_dump_json(ensure_ascii=False)
    plan: Plan = state.get("plan")
    plan_text = plan.model_dump_json(ensure_ascii=False) if plan else "（暂无计划）"
    quiz: Quiz = state.get("quiz")
    quiz_text = quiz.model_dump_json(ensure_ascii=False) if quiz else "（暂无测验）"

    messages = (
        [SystemMessage(content=_REVIEW_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [
            HumanMessage(
                content=f"【学生画像】\n{profile_json}\n\n"
                f"【学习计划】\n{plan_text}\n\n"
                f"【自测题】\n{quiz_text}\n\n请输出学情复盘 JSON。"
            )
        ]
    )
    review: Review = chat.invoke(messages)
    return {"review": review}

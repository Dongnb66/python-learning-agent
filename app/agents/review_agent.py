"""ReviewAgent —— 学情复盘与下一步建议。

跨会话记忆的出口：若 load_memory 读到了上次学情，
本智能体会额外输出「上次 vs 本次」的纵向对比（comparison 字段）。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, LearningSession, Plan, Quiz, Review

_REVIEW_SYSTEM = """你是「学情复盘智能体」。
结合学生画像、学习计划与自测题，输出复盘报告：
- mastery：对学生当前掌握度的总体判断；
- strengths：已掌握的亮点（2~3 条）；
- gaps：仍存在的薄弱项（2~3 条）；
- suggestions：下一步学习建议（2~4 条）；
- comparison：若提供了【上次学情】，用 1~2 句话说明「上次 vs 本次」的变化
  （哪些薄弱点已补齐、哪些仍未改善）；若没有上次学情，该字段填空字符串。
输出符合 Review schema 的 JSON。"""


def _render_memory(memory: LearningSession | None) -> str:
    """把上次学情渲染成提示词片段。没有历史时明确告知，避免模型编造对比。"""
    if memory is None:
        return "【上次学情】无（该学员为首次学习，comparison 请填空字符串）"
    return (
        f"【上次学情（{memory.created_at}）】\n"
        f"目标：{memory.goal or '未记录'}\n"
        f"掌握度：{memory.mastery or '未记录'}\n"
        f"亮点：{'、'.join(memory.strengths) or '无'}\n"
        f"薄弱项：{'、'.join(memory.gaps) or '无'}\n"
        f"当时的建议：{'、'.join(memory.suggestions) or '无'}"
    )


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
                f"{_render_memory(state.get('memory'))}\n\n"
                f"【学习计划】\n{plan_text}\n\n"
                f"【自测题】\n{quiz_text}\n\n请输出学情复盘 JSON。"
            )
        ]
    )
    review: Review = chat.invoke(messages)
    return {"review": review}

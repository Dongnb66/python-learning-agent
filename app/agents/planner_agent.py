"""PlannerAgent —— 基于画像生成个性化学习计划。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Plan

_PLANNER_SYSTEM = """你是「学习计划规划智能体」。
输入：学生的 6 维画像（JSON）。
输出：符合 Plan schema 的学习计划。要求：
- goal 用一句话概括总目标；
- steps 是 3~6 个有序步骤，每步 title 简短、description 说明做什么、est_minutes 估计耗时；
- total_minutes 为所有步骤耗时之和；
- 计划要针对画像中的薄弱点与目标，难度循序渐进。"""


def build_plan_node(state: AgentState) -> dict:
    chat = get_structured_model(Plan, temperature=0.3)
    profile_json = state["profile"].model_dump_json(ensure_ascii=False)
    messages = (
        [SystemMessage(content=_PLANNER_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [HumanMessage(content=f"学生画像如下，请生成学习计划：\n{profile_json}")]
    )
    plan: Plan = chat.invoke(messages)
    return {"plan": plan}

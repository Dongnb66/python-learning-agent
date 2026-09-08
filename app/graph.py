"""LangGraph 状态图：把 5 个智能体串成多智能体工作流。

执行顺序：
    profile → planner → resource → quiz → review
每个节点只更新自己负责的字段，最终状态汇聚全部产物。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.planner_agent import build_plan_node
from app.agents.profile_agent import build_profile_node
from app.agents.quiz_agent import build_quiz_node
from app.agents.resource_agent import build_resources_node
from app.agents.review_agent import build_review_node
from app.models import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("profile", build_profile_node)
    g.add_node("planner", build_plan_node)
    g.add_node("resource", build_resources_node)
    g.add_node("quiz", build_quiz_node)
    g.add_node("review", build_review_node)

    g.set_entry_point("profile")
    g.add_edge("profile", "planner")
    g.add_edge("planner", "resource")
    g.add_edge("resource", "quiz")
    g.add_edge("quiz", "review")
    g.add_edge("review", END)
    return g.compile()

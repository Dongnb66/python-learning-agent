"""LangGraph 状态图：7 节点多智能体学习闭环。

执行顺序：

    load_memory → profile → planner → resource → quiz → review → save_memory

其中：
- 5 个节点是 LLM 智能体（profile / planner / resource / quiz / review）；
- 2 个节点是记忆节点（load_memory / save_memory），纯 IO、不调 LLM，
  负责把「长期画像 + 学情轨迹」读出 / 写回 SQLite，
  让同一个学生第二次来学习时能拿到上次的结果并做纵向对比。

每个节点只更新自己负责的字段，最终状态汇聚全部产物。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.memory_agent import build_load_memory_node, build_save_memory_node
from app.agents.planner_agent import build_plan_node
from app.agents.profile_agent import build_profile_node
from app.agents.quiz_agent import build_quiz_node
from app.agents.resource_agent import build_resources_node
from app.agents.review_agent import build_review_node
from app.models import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_memory", build_load_memory_node)
    g.add_node("profile", build_profile_node)
    g.add_node("planner", build_plan_node)
    g.add_node("resource", build_resources_node)
    g.add_node("quiz", build_quiz_node)
    g.add_node("review", build_review_node)
    g.add_node("save_memory", build_save_memory_node)

    g.set_entry_point("load_memory")
    g.add_edge("load_memory", "profile")
    g.add_edge("profile", "planner")
    g.add_edge("planner", "resource")
    g.add_edge("resource", "quiz")
    g.add_edge("quiz", "review")
    g.add_edge("review", "save_memory")
    g.add_edge("save_memory", END)
    return g.compile()

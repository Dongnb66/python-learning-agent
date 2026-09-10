"""LangGraph 状态图：多智能体学习闭环 + 自主辅导循环。

执行路径
--------

    load_memory → profile → planner → resource → quiz → review
                                                        │
                                        route_after_review（条件边）
                                              ┌─────────┴─────────┐
                                    有薄弱项 ↓                   ↓ 无薄弱项
                                           tutor             save_memory
                                              │                   │
                                              └────→ save_memory ←┘
                                                        │
                                                       END

节点职责
--------
- 5 个 LLM 智能体节点：profile（画像）/ planner（规划）/ resource（RAG 资源）/
  quiz（自测）/ review（复盘）——单次调用型，执行路径由代码固定（Workflow）。
- 2 个纯 IO 记忆节点：load_memory / save_memory——不调 LLM，负责跨会话记忆读写，
  且做故障隔离，异常不中断主流程。
- 1 个自主决策节点：**tutor（自主辅导）**——ReAct 式工具调用循环，
  调哪些工具、调几轮、何时停止，全部由模型在运行时决定（Agent）。

设计取舍（面试可讲）
------------------
为什么其余节点仍是确定性 Workflow，只把 tutor 做成 Agent？
教学主链路（画像→计划→资源→测验→复盘）需要**可复现、可审计、成本可控**——
同一份输入必须给出同一份教学路径，所以固定编排更合适。
而「这个学生到底卡在哪」是一个**开放探索问题**：查什么、查几次事先无法枚举，
必须让模型自主决定。**该确定的地方确定，该自主的地方自主**，才是合理设计。
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.memory_agent import build_load_memory_node, build_save_memory_node
from app.agents.planner_agent import build_plan_node
from app.agents.profile_agent import build_profile_node
from app.agents.quiz_agent import build_quiz_node
from app.agents.resource_agent import build_resources_node
from app.agents.review_agent import build_review_node
from app.agents.tutor_agent import build_tutor_node, route_after_review
from app.models import AgentState


def build_graph():
    g = StateGraph(AgentState)

    # ---- 记忆节点（纯 IO）----
    g.add_node("load_memory", build_load_memory_node)
    g.add_node("save_memory", build_save_memory_node)

    # ---- 5 个 LLM 智能体（单次调用型）----
    g.add_node("profile", build_profile_node)
    g.add_node("planner", build_plan_node)
    g.add_node("resource", build_resources_node)
    g.add_node("quiz", build_quiz_node)
    g.add_node("review", build_review_node)

    # ---- 自主决策节点（ReAct 循环）----
    g.add_node("tutor", build_tutor_node)

    g.set_entry_point("load_memory")
    g.add_edge("load_memory", "profile")
    g.add_edge("profile", "planner")
    g.add_edge("planner", "resource")
    g.add_edge("resource", "quiz")
    g.add_edge("quiz", "review")

    # 条件边：复盘后自主判断是否需要追加辅导，而不是无条件往下走
    g.add_conditional_edges(
        "review",
        route_after_review,
        {"tutor": "tutor", "save_memory": "save_memory"},
    )

    g.add_edge("tutor", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile()

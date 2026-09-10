"""记忆节点 —— 跨会话记忆的读写两端。

管线形态（7 节点）：

    load_memory → profile → planner → resource → quiz → review → save_memory

- load_memory：管线最前端，读出该用户的历史会话数与最近一次学情，注入共享状态；
- save_memory：管线最末端，把本次画像与复盘结论写回 SQLite，成为下一次的「历史」。

设计要点（面试可讲）：
1. **两层持久化**：profiles 存长期画像（这个学生是谁），learning_sessions 存学情轨迹
   （这个学生历次学得怎么样）——合起来才是完整的跨会话记忆，缺一层都撑不起纵向对比。
2. **故障隔离**：记忆读写失败只往 errors 里记一笔，绝不抛异常中断主流程。
   在 LLM 应用里"记忆挂了导致整个服务不可用"是不可接受的设计。
3. **无 LLM 调用**：这两个节点是纯 IO，不消耗 token，也不受模型波动影响。
"""
from __future__ import annotations

from app import db
from app.models import AgentState


def build_load_memory_node(state: AgentState) -> dict:
    """读出历史学情，供下游 review 智能体做「上次 vs 本次」对比。

    新学员（memory_hits == 0）时 memory 为 None，下游自动跳过对比段落。
    """
    user_id = state.get("user_id") or ""
    errors = list(state.get("errors") or [])

    try:
        db.init_db()
        hits = db.count_sessions(user_id)
        memory = db.load_last_session(user_id)
    except Exception as exc:  # 记忆故障隔离：不阻塞主流程
        errors.append(f"load_memory 失败（已跳过）：{type(exc).__name__}: {exc}")
        return {"memory": None, "memory_hits": 0, "errors": errors}

    return {"memory": memory, "memory_hits": hits, "errors": errors}


def build_save_memory_node(state: AgentState) -> dict:
    """把本次画像与复盘结论写回，形成下一轮的历史。"""
    user_id = state.get("user_id") or ""
    errors = list(state.get("errors") or [])

    try:
        db.init_db()
        profile = state.get("profile")
        if profile is not None:
            db.save_profile(user_id, profile)

        review = state.get("review")
        if review is not None:
            plan = state.get("plan")
            db.save_session(user_id, review, goal=getattr(plan, "goal", "") or "")
    except Exception as exc:  # 同上：写记忆失败不影响本次已产出的结果
        errors.append(f"save_memory 失败（已跳过）：{type(exc).__name__}: {exc}")

    return {"errors": errors}

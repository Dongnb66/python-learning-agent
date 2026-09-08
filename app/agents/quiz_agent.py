"""QuizAgent —— 针对学习计划生成自测题。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._common import to_lc_messages
from app.llm import get_structured_model
from app.models import AgentState, Plan, Quiz, QuizQuestion

_QUIZ_SYSTEM = """你是「自测题生成智能体」。
根据学生的学习计划与画像，生成 3 道检测题，覆盖关键知识点。
要求：
- 优先出选择题（options 2~4 项），也可出简答；
- answer 给出正确答案；
- explanation 用 1~2 句说明原理；
- 输出符合 Quiz schema 的 JSON（questions 列表）。"""


def build_quiz_node(state: AgentState) -> dict:
    chat = get_structured_model(Quiz, temperature=0.3)
    plan: Plan = state.get("plan")
    plan_text = plan.model_dump_json(ensure_ascii=False) if plan else "（暂无计划）"
    profile_json = state["profile"].model_dump_json(ensure_ascii=False)

    messages = (
        [SystemMessage(content=_QUIZ_SYSTEM)]
        + to_lc_messages(state["messages"])
        + [
            HumanMessage(
                content=f"【学生画像】\n{profile_json}\n\n"
                f"【学习计划】\n{plan_text}\n\n请生成自测题 JSON。"
            )
        ]
    )
    quiz: Quiz = chat.invoke(messages)
    # 兜底：保证字段存在
    quiz.questions = [QuizQuestion(**q.model_dump()) for q in quiz.questions]
    return {"quiz": quiz}

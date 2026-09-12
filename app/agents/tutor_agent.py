"""TutorAgent —— 自主辅导智能体（ReAct 式工具调用循环）。

与前 5 个智能体的本质区别
--------------------------
前 5 个节点（profile / planner / resource / quiz / review）是**单次调用型**：
一次 prompt 进、一份结构化结果出，执行路径完全由代码固定。

本节点是**自主决策型**（Agent，而非 Workflow）：

    ┌─────────────┐
    │  LLM 决策    │ ← 看当前已知信息，自己决定「下一步要不要查、查什么」
    └──────┬──────┘
           │ 产出 tool_calls（可 0~N 个）
           ▼
    ┌─────────────┐
    │  执行工具    │  只读查询：检索资料 / 读画像 / 读学情 / 读技能正文
    └──────┬──────┘
           │ observation 回灌上下文
           ▼
       回到决策（最多 max_rounds 轮）

循环次数、调哪些工具、何时停止，**全部由模型在运行时决定**，
代码只提供工具集合与轮数上限（防止无限循环 / 成本失控）。
每轮的决策与观察都记进 trace，形成可审计的决策轨迹。

降级策略（工程健壮性）
--------------------
- 未配置 LLM Key，或 `TUTOR_MODE=policy`：走**规则策略**——同样经过
  「决策 → 调工具 → 观察 → 再决策」的循环结构，只是决策由启发式规则给出，
  保证 Demo / 单测在零 Key 环境下依然能完整跑通；
- LLM 调用异常：捕获后回落到规则策略，并把原因记入 errors，绝不中断主流程。
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents._common import to_lc_messages
from app import config
from app import skills as skills_registry
from app import telemetry
from app.config import get_settings
from app.models import AgentState, TutorSession, ToolCallRecord
from app.tools import TOOL_REGISTRY

# ReAct 循环上限：防止模型陷入无限工具调用
MAX_ROUNDS = 4

# 这些工具的用户标识必须以服务端状态为准，不信任模型自填的参数
_USER_SCOPED_TOOLS = {"get_student_profile", "get_last_session", "count_history_sessions"}

# 明显是占位符的 Key，视为「未配置」，避免每次白跑一次注定 401 的请求。
# 实现已收敛到配置层（app.config.has_usable_llm_key），这里只做转发，
# 避免「两处各一套占位符规则、改一处漏一处」。
_PLACEHOLDER_MARKERS = config._PLACEHOLDER_MARKERS


def _has_usable_key(api_key: str | None) -> bool:
    """判断是否配置了「看起来可用」的 LLM Key（见 app.config.has_usable_llm_key）。"""
    return config.has_usable_llm_key(api_key)

_SYSTEM = """你是「自主辅导智能体」，负责在学生完成一轮学习后，主动排查他真正卡在哪里。

你可以调用工具来获取事实依据，而不是凭记忆猜测：
- count_history_sessions(user_id)：判断他是新学员还是老学员；
- get_last_session(user_id)：看上次的掌握度、薄弱项和当时的建议；
- get_student_profile(user_id)：看长期画像（知识基础 / 认知风格 / 资源偏好）；
- search_learning_resources(query)：在真实资料库里检索可用于补短板的学习材料；
- load_skill(skill_name)：读取某个技能的完整操作步骤（技能目录见下方「可用技能」清单）。

**决策规则（自己判断，不要机械照搬）**：
1. 先确认信息是否足够。信息不足就调工具去查；已经够了就直接给结论，不要为了凑数而调用工具。
2. 老学员要对照【上次学情】看哪些薄弱项没改善；新学员则不必做纵向对比。
3. 检索资料时，query 直接使用具体的薄弱知识点（例如「递归」「指针」），不要用「学习建议」这类空泛词。
4. 资料库检索不到的内容，如实说明「暂无现成材料」，**严禁编造资料名或链接**。
5. 若「可用技能」清单里有技能命中当前场景，先 load_skill 读取完整步骤、按指南辅导；
   清单里没有的技能不要假设它存在，更不要编造步骤。

信息收集充分后，停止调用工具，直接输出一份简短的辅导结论（120 字以内）：
先说卡点是什么、再说下一步具体该做什么（可引用检索到的真实资料标题）。"""


# --------------------------------------------------------------------------- #
# 工具执行器
# --------------------------------------------------------------------------- #
def _execute_tool(name: str, args: dict[str, Any], user_id: str) -> str:
    """按名执行工具，返回给模型看的观察文本。

    安全设计：涉及用户数据的工具强制注入服务端已知的 user_id，
    避免模型传入错误甚至伪造的标识去读取他人的画像 / 学情。
    """
    telemetry.bump(telemetry.C_TOOL_CALLS)
    tool_obj = TOOL_REGISTRY.get(name)
    if tool_obj is None:
        available = "、".join(TOOL_REGISTRY)
        return f"（未知工具 {name}，可用工具：{available}）"

    safe_args = dict(args or {})
    if name in _USER_SCOPED_TOOLS:
        safe_args["user_id"] = user_id

    try:
        with telemetry.span(f"tool.{name}"):
            return str(tool_obj.invoke(safe_args))
    except Exception as exc:  # 单个工具失败不该让整轮循环崩掉
        return f"（工具 {name} 执行失败：{type(exc).__name__}: {exc}）"


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# --------------------------------------------------------------------------- #
# 路径 A：真实模型驱动的 ReAct 循环
# --------------------------------------------------------------------------- #
def _run_llm_react(question: str, user_id: str, history: list) -> TutorSession:
    from app.llm import get_chat_model

    llm = get_chat_model(temperature=0.2).bind_tools(list(TOOL_REGISTRY.values()))

    # 技能懒加载：常驻上下文的只有「清单」（元数据），正文必须经 load_skill 按需读取。
    # 每次进入节点时重新渲染，保证技能文件热更新后无需重启进程。
    system_prompt = _SYSTEM + skills_registry.skills_metadata_prompt()

    messages: list[Any] = (
        [SystemMessage(content=system_prompt)]
        + to_lc_messages(history)
        + [HumanMessage(content=question)]
    )

    trace: list[ToolCallRecord] = []
    used: list[str] = []
    rounds = 0

    for rnd in range(1, MAX_ROUNDS + 1):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        tool_calls = list(getattr(ai, "tool_calls", None) or [])
        if not tool_calls:
            # 模型自主判断「信息已足够」→ 停止循环
            return TutorSession(
                question=question,
                answer=str(ai.content or "").strip() or "（模型未给出结论）",
                tools_used=_dedup(used),
                rounds=rounds,
                mode="llm",
                trace=trace,
            )

        rounds = rnd
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            observation = _execute_tool(name, args, user_id)
            used.append(name)
            trace.append(
                ToolCallRecord(
                    round=rnd,
                    tool=name,
                    args=json.dumps(args, ensure_ascii=False),
                    observation=observation[:200],
                )
            )
            messages.append(
                ToolMessage(content=observation, tool_call_id=call.get("id") or f"call-{rnd}")
            )

    # 达到轮数上限仍未收敛：强制收口，避免成本不可控
    messages.append(
        HumanMessage(content="已达到工具调用轮数上限，请基于以上已获取的信息直接给出结论。")
    )
    final = llm.invoke(messages)
    return TutorSession(
        question=question,
        answer=str(getattr(final, "content", "") or "").strip() or "（已达轮数上限）",
        tools_used=_dedup(used),
        rounds=rounds,
        mode="llm",
        trace=trace,
    )


# --------------------------------------------------------------------------- #
# 路径 B：规则策略（无 Key / 模型异常时的兜底）
# --------------------------------------------------------------------------- #
def _run_policy_react(question: str, user_id: str, hints: dict[str, Any]) -> TutorSession:
    """与 LLM 路径同构的「决策 → 调工具 → 观察 → 再决策」循环，决策改由规则给出。

    规则策略同样会真实执行工具（读 DB、跑 BM25），因此不是空壳；
    只是「下一步调什么」由启发式规则决定，而非模型。
    """
    trace: list[ToolCallRecord] = []
    used: list[str] = []
    observations: dict[str, str] = {}
    rounds = 0

    def act(name: str, args: dict[str, Any]) -> str:
        """执行一次「决策 → 调工具 → 观察」，轮次自增。"""
        nonlocal rounds
        rounds += 1
        obs = _execute_tool(name, args, user_id)
        used.append(name)
        observations[name] = obs
        trace.append(
            ToolCallRecord(
                round=rounds,
                tool=name,
                args=json.dumps(args, ensure_ascii=False),
                observation=obs[:200],
            )
        )
        return obs

    # 决策 1：新学员还是老学员？（决定后续要不要做纵向对比）
    act("count_history_sessions", {"user_id": user_id})
    digits = "".join(ch for ch in observations["count_history_sessions"] if ch.isdigit())
    is_returning = bool(digits) and int(digits) > 0

    # 决策 2：老学员才读上次学情，避免新学员做无意义的纵向对比
    if is_returning:
        act("get_last_session", {"user_id": user_id})

    # 决策 3：拿薄弱点去真实资料库检索可用的补强材料
    weak = (hints.get("gaps") or [])[:2]
    if not weak:
        weak = [hints.get("weak_point") or "学习方法"]
    query = " ".join(str(w) for w in weak if w).strip() or "学习方法"
    act("search_learning_resources", {"query": query})

    # 决策 4：补读长期画像（资源偏好 / 认知风格），让建议更贴合个体
    act("get_student_profile", {"user_id": user_id})

    gaps_text = "、".join(str(g) for g in weak) or "暂未识别出明确薄弱点"
    resource_text = observations.get("search_learning_resources", "")
    has_material = "未找到匹配资料" not in resource_text
    memory_text = observations.get("get_last_session", "")

    parts = [f"当前主要卡点：{gaps_text}。"]
    if is_returning and memory_text:
        parts.append(f"对照历史学情：{memory_text[:120]}")
    if has_material:
        first_line = next(
            (ln for ln in resource_text.splitlines() if ln.strip().startswith("-")), ""
        )
        parts.append(f"建议先补：{first_line.lstrip('- ').strip()[:120]}")
    else:
        parts.append("资料库暂无该主题的现成材料，建议如实告知学生并给出通用的学习路径建议。")

    return TutorSession(
        question=question,
        answer=" ".join(parts),
        tools_used=_dedup(used),
        rounds=rounds,
        mode="policy",
        trace=trace,
    )


# --------------------------------------------------------------------------- #
# 节点入口
# --------------------------------------------------------------------------- #
def _build_diagnostic_question(state: AgentState) -> str:
    """把本轮复盘结果转成一个诊断性问题，作为自主循环的起点。"""
    review = state.get("review")
    gaps = list(getattr(review, "gaps", []) or [])
    mastery = getattr(review, "mastery", "") or "未评估"
    gap_text = "、".join(gaps) if gaps else "未明确"

    return (
        f"该学生本轮掌握度：{mastery}；复盘识别出的薄弱项：{gap_text}。\n"
        "请自主判断需要查询哪些信息，找出他真正的卡点，并给出下一步的具体行动建议。"
    )


def build_tutor_node(state: AgentState) -> dict:
    """自主辅导节点：以 ReAct 循环收集事实依据后给出辅导结论。"""
    user_id = state.get("user_id") or ""
    errors = list(state.get("errors") or [])
    question = _build_diagnostic_question(state)

    review = state.get("review")
    hints = {
        "gaps": list(getattr(review, "gaps", []) or []),
        "weak_point": getattr(getattr(state.get("profile"), "weakPoints", None), "details", "")
        or "",
    }

    # 工具要读 DB，先确保表结构就绪（create_all 幂等）；失败不阻塞——
    # 工具内部会各自返回可读的失败信息，由自主循环决定如何处理。
    try:
        from app import db

        db.init_db()
    except Exception as exc:
        errors.append(f"tutor 初始化数据库失败（工具将返回失败信息）：{type(exc).__name__}: {exc}")

    settings = get_settings()
    mode = (os.getenv("TUTOR_MODE") or "auto").strip().lower()

    try:
        if mode == "policy" or (mode == "auto" and not _has_usable_key(settings.llm_api_key)):
            session = _run_policy_react(question, user_id, hints)
        else:
            session = _run_llm_react(question, user_id, list(state.get("messages") or []))
        # 模型没给出结论时退回规则策略，保证节点一定有产出
        if not session.answer.strip():
            session = _run_policy_react(question, user_id, hints)
    except Exception as exc:  # 自主循环整体失败 → 降级，不阻塞管线
        errors.append(f"tutor 自主循环失败（已降级规则策略）：{type(exc).__name__}: {exc}")
        try:
            session = _run_policy_react(question, user_id, hints)
        except Exception as exc2:  # 极端情况：连规则策略也失败
            errors.append(f"tutor 规则策略亦失败（已跳过）：{type(exc2).__name__}: {exc2}")
            return {"tutoring": None, "agent_trace": [], "errors": errors}

    return {"tutoring": session, "agent_trace": session.trace, "errors": errors}


def route_after_review(state: AgentState) -> str:
    """复盘之后的条件边：自主决定「是否需要追加一对一辅导」。

    判定依据来自复盘智能体的产出——只要有明确薄弱项，就进入辅导循环；
    没有薄弱项说明本轮目标已达成，直接收尾，避免无谓的模型开销。
    """
    review = state.get("review")
    gaps = list(getattr(review, "gaps", []) or [])
    return "tutor" if gaps else "save_memory"

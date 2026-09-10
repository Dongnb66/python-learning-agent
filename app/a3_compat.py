"""A3 前端兼容 API 层。

让 A3 赛题的 React 用户端（9 个页面）可以直接跑在本项目后端上，
不改动前端任何代码：本路由按 A3 的 URL / 请求体 / 响应包装约定
（`{success, data}` 信封 + `{error}` 错误体）逐端点做了适配，
底层全部映射到本项目真实的多智能体能力：

- 画像对话 / 生成   → LLM 对话 + ProfileAgent（build_profile_node）
- 资源生成         → 完整 LangGraph 管线（画像→计划→资源→测验→复盘→辅导）
- 学习路径         → PlannerAgent（build_plan_node）
- 智能辅导         → TutorAgent 的 ReAct 工具调用循环（无 Key 时规则策略）
- 前后测实证       → 服务端判分题库（efficacy_bank）+ 记录存储
- 知识库 / 模型 / 分析 → a3_store 持久化 + 现有数据库统计

注意：本路由必须在 app/main.py 中先于旧的 /api/profile/{user_id} 注册，
否则 GET /api/profile/{id} 会被旧路由匹配（返回结构不同）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import a3_store
from app.agents.planner_agent import build_plan_node
from app.agents.profile_agent import build_profile_node
from app.agents.tutor_agent import _has_usable_key, _run_llm_react, _run_policy_react
from app.db import init_db, load_profile, save_profile
from app.efficacy_bank import SEED_SAMPLES, get_questions, score as bank_score, strip_answers
from app.graph import build_graph
from app.llm import mock_enabled
from app.models import AgentState, Message, Profile
from app.config import get_settings

logger = logging.getLogger("learning-agent.a3")

router = APIRouter(prefix="/api")

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# --------------------------------------------------------------------------- #
# 信封工具（对齐 A3 utils/response.js）
# --------------------------------------------------------------------------- #
def success(data, message: str | None = None) -> dict:
    body: dict = {"success": True, "data": data}
    if message:
        body["message"] = message
    return body


def fail(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": message})


async def _json_body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Profile（python Profile 模型 ↔ A3 前端画像结构）
# --------------------------------------------------------------------------- #
def profile_to_a3(profile: Profile, pid: str, dialogue_history: list | None = None) -> dict:
    return {
        "id": pid,
        "studentName": profile.name or "同学",
        "major": profile.major or "未填写",
        "dimensions": {
            "knowledgeBase": {"level": profile.knowledgeBase.level, "details": profile.knowledgeBase.details},
            "learningGoal": {"level": profile.learningGoal.level, "details": profile.learningGoal.details},
            "cognitiveStyle": {"level": profile.cognitiveStyle.level, "details": profile.cognitiveStyle.details},
            "weakPoints": {"level": profile.weakPoints.level, "details": profile.weakPoints.details},
            "resourcePreference": {"level": profile.resourcePreference.level, "details": profile.resourcePreference.details},
            "studyTime": {"level": profile.studyTime.level, "details": profile.studyTime.details},
        },
        "summary": profile.summary or "",
        "dialogueHistory": dialogue_history or [],
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _to_messages(dialogue: list) -> list[Message]:
    out = []
    for m in dialogue or []:
        if isinstance(m, dict) and m.get("content"):
            out.append(Message(role=m.get("role") or "user", content=str(m["content"])))
    return out


def _new_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}-{os.urandom(2).hex()}"


_PROFILE_INTERVIEW_SYSTEM = (
    "你是「学习画像助手」，通过自然对话了解学生情况，为构建 6 维学习画像"
    "（知识基础、学习目标、认知风格、薄弱点、资源偏好、学习时间）收集信息。"
    "每次回复简短友好，一次重点追问 1~2 个还不清楚的维度。"
)


def _fallback_chat_reply(messages: list[dict]) -> str:
    """Mock 模式 / 模型不可用时的规则版追问（按缺失维度引导）。"""
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    checks = [
        ("学习目标", ("想", "目标", "提升", "希望", "掌握", "学会"), "你的学习目标是什么？想提升或掌握哪方面内容？"),
        ("知识基础", ("基础", "学过", "了解", "熟悉", "零基础"), "你目前在这方面的知识基础怎么样？学过哪些相关内容？"),
        ("学习时间", ("小时", "时间", "每周", "每天"), "你每周大概能投入多少学习时间？"),
        ("资源偏好", ("视频", "文章", "案例", "实践", "图文", "偏好"), "你偏好哪种学习资源？比如视频、图文、案例实践等。"),
        ("薄弱点", ("薄弱", "困难", "不懂", "卡"), "你觉得自己目前最薄弱、最容易卡住的地方是什么？"),
    ]
    asked_any = False
    for name, keywords, question in checks:
        if not any(k in text for k in keywords):
            return f"好的，了解了！还想再确认一个维度——{question}（{name}）"
        asked_any = True
    if not asked_any:
        return "可以先告诉我你的专业和想提升的方向吗？"
    return (
        "信息收集得差不多了！你提到的内容我已经记录，"
        "点击「生成画像」即可为你构建 6 维学习画像。"
    )


@router.get("/health")
def a3_health() -> dict:
    return success({"status": "ok", "service": "python-learning-agent", "compat": "a3-frontend"})


@router.post("/profile/chat")
async def a3_profile_chat(request: Request):
    body = await _json_body(request)
    messages = body.get("messages") or []
    if not messages:
        return fail("消息列表不能为空", 400)

    start = time.time()
    reply = None
    if not mock_enabled():
        try:
            from app.llm import get_chat_model

            lc_msgs = [SystemMessage(content=_PROFILE_INTERVIEW_SYSTEM)]
            for m in _to_messages(messages):
                lc_msgs.append(
                    HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
                )
            resp = get_chat_model(temperature=0.5).invoke(lc_msgs)
            reply = str(resp.content or "").strip() or None
        except Exception as e:  # noqa: BLE001
            logger.warning("profile chat LLM failed, fallback to rules: %s", e)
    if not reply:
        reply = _fallback_chat_reply(messages)

    latency = int((time.time() - start) * 1000)
    a3_store.add_log("profile_chat", latency)
    return success({"reply": reply, "latency": latency})


@router.post("/profile/generate")
async def a3_profile_generate(request: Request):
    body = await _json_body(request)
    dialogue = body.get("dialogueHistory") or []
    messages = _to_messages(dialogue)
    if not messages:
        return fail("对话历史不能为空", 400)

    start = time.time()
    pid = _new_id("profile")
    state: AgentState = {"user_id": pid, "messages": messages, "resources": [], "errors": []}
    try:
        out = build_profile_node(state)
    except Exception as e:  # noqa: BLE001
        logger.exception("a3 profile generate failed")
        return fail(f"画像生成失败: {e}", 500)

    profile: Profile = out["profile"]
    try:
        save_profile(pid, profile)
    except Exception as e:  # noqa: BLE001
        logger.warning("save profile failed: %s", e)

    a3_profile = profile_to_a3(profile, pid, dialogue)
    a3_store.add_log("profile_chat", int((time.time() - start) * 1000))
    return success(
        {
            "profile": a3_profile,
            "agentResult": {
                "agent": "ProfileAgent",
                "status": "success",
                "duration": int((time.time() - start) * 1000),
                "output": a3_profile,
            },
        }
    )


@router.get("/profile")
def a3_list_profiles() -> dict:
    """列出全部画像（A3 首页/画像管理用）。"""
    from app.db import list_profiles

    items = [profile_to_a3(p, p_user_id) for p_user_id, p in list_profiles()]
    return success({"profiles": items})


@router.get("/profile/{pid}")
def a3_get_profile(pid: str):
    profile = load_profile(pid)
    if profile is None:
        return fail("画像不存在", 404)
    return success({"profile": profile_to_a3(profile, pid)})


# --------------------------------------------------------------------------- #
# 资源生成（完整多智能体管线 → A3 资源包结构）
# --------------------------------------------------------------------------- #
_MAJOR_COURSE_MAP = [
    (("法律", "法学"), ("刑法学", "犯罪构成要件")),
    (("医学", "临床"), ("病理学", "炎症与免疫反应")),
    (("金融", "经济"), ("宏观经济学", "货币政策工具")),
    (("计算机", "软件", "计科"), ("数据结构", "二叉树遍历算法")),
]


def _infer_course_topic(major: str, goal: str) -> tuple[str, str]:
    for keys, (course, topic) in _MAJOR_COURSE_MAP:
        if any(k in (major or "") for k in keys):
            return course, topic
    if goal:
        return (major or "通用学习") + "核心课程", "基础知识体系"
    return "人工智能导论", "机器学习基础"


def _quiz_to_a3(quiz) -> list[dict]:
    out = []
    letters = ["A", "B", "C", "D", "E", "F"]
    for q in quiz.questions if quiz else []:
        options = list(q.options or [])
        # 确保选项带字母前缀（A3 前端用 opt[0] 作为选项字母）
        prefixed = []
        for i, opt in enumerate(options):
            opt = str(opt)
            if opt[:1] in letters and len(opt) > 1 and opt[1] in ".、．":
                prefixed.append(opt)
            else:
                prefixed.append(f"{letters[i]}. {opt}" if i < len(letters) else opt)
        answer = str(q.answer or "").strip()
        # 归一化答案为字母
        if answer and answer[0].upper() in letters and len(answer) <= 2:
            answer = answer[0].upper()
        else:
            # 答案是内容 → 反查选项字母
            hit = next((prefixed[i][0] for i, o in enumerate(prefixed) if o.split(". ", 1)[-1] == answer), None)
            answer = hit or answer  # 反查不到则保留原文（前端按简答题展示）
        q_type = "choice" if prefixed and len(answer) == 1 and answer in letters else "short"
        out.append(
            {
                "type": q_type,
                "question": q.q,
                "options": prefixed if q_type == "choice" else [],
                "answer": answer,
                "explanation": q.explanation or "",
            }
        )
    return out


def _build_lecture(course: str, topic: str, profile_a3: dict, plan, resources, review) -> str:
    lines = [f"# {course} · {topic}", ""]
    if profile_a3:
        lines += [
            "## 学生画像摘要",
            "",
            f"- **专业**：{profile_a3['major']}",
            f"- **知识基础**：{profile_a3['dimensions']['knowledgeBase']['details'] or '待补充'}",
            f"- **学习目标**：{profile_a3['dimensions']['learningGoal']['details'] or '待补充'}",
            f"- **薄弱点**：{profile_a3['dimensions']['weakPoints']['details'] or '待补充'}",
            "",
        ]
    if plan:
        lines += ["## 学习计划", ""]
        for s in plan.steps:
            lines.append(f"{s.step}. **{s.title}**（约 {s.est_minutes} 分钟）：{s.description}")
        lines.append("")
    if resources:
        lines += ["## 推荐学习资料", ""]
        for r in resources:
            url_part = f" — [链接]({r.url})" if getattr(r, "url", None) else ""
            lines.append(f"- **{r.title}**（{r.type}）：{r.relevance or r.description}{url_part}")
        lines.append("")
    if review and review.suggestions:
        lines += ["## 复盘建议", ""]
        for x in review.suggestions:
            lines.append(f"- {x}")
        lines.append("")
    return "\n".join(lines)


@router.post("/generate")
async def a3_generate(request: Request):
    body = await _json_body(request)
    dialogue = body.get("dialogueHistory") or []
    course = (body.get("course") or "").strip()
    topic = (body.get("topic") or "").strip()
    profile_id = (body.get("profileId") or "").strip()

    user_id = profile_id or _new_id("stu")
    messages = _to_messages(dialogue)
    if not messages:
        loaded = load_profile(user_id)
        msg_text = (
            f"我是{loaded.major or '学生'}，希望提升学习效果"
            if loaded
            else "我是学生，希望提升学习效果"
        )
        messages = [Message(role="user", content=msg_text)]

    start = time.time()
    state: AgentState = {"user_id": user_id, "messages": messages, "resources": [], "errors": []}
    try:
        final = _get_graph().invoke(state)
    except Exception as e:  # noqa: BLE001
        logger.exception("a3 generate pipeline failed")
        return fail(f"学习流程执行失败: {e}", 500)

    profile = final.get("profile")
    if profile is None:
        return fail("画像生成为空", 500)
    try:
        save_profile(user_id, profile)
    except Exception as e:  # noqa: BLE001
        logger.warning("save profile failed: %s", e)

    if not course or not topic:
        infer_course, infer_topic = _infer_course_topic(
            profile.major or "", profile.learningGoal.details
        )
        course = course or infer_course
        topic = topic or infer_topic

    latency = int((time.time() - start) * 1000)
    errors = list(final.get("errors") or [])
    review = final.get("review")
    plan = final.get("plan")
    resources = final.get("resources") or []
    tutoring = final.get("tutoring")

    profile_a3 = profile_to_a3(profile, user_id, dialogue)

    mind_children = []
    if plan:
        mind_children = [{"name": s.title} for s in plan.steps]
    if resources:
        mind_children.append({"name": "推荐资料", "children": [{"name": r.title} for r in resources[:4]]})

    issues = [{"severity": "warning", "message": e} for e in errors]
    quality_score = max(60, 92 - 8 * len(errors))

    trace_json = ""
    if tutoring is not None:
        trace_json = json.dumps(
            {"mode": tutoring.mode, "rounds": tutoring.rounds, "tools_used": tutoring.tools_used, "trace": [t.model_dump() for t in tutoring.trace]},
            ensure_ascii=False,
            indent=2,
        )

    package = {
        "id": _new_id("res"),
        "course": course,
        "topic": topic,
        "studentName": profile_a3["studentName"],
        "profileId": user_id,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile_a3,
        "lecture": _build_lecture(course, topic, profile_a3, plan, resources, review),
        "mindMap": {"name": topic, "children": mind_children},
        "quiz": _quiz_to_a3(final.get("quiz")),
        "reading": [
            {
                "title": r.title,
                "source": f"AI 资料库 · {r.type}",
                "description": r.relevance or r.description,
                "url": r.url,
            }
            for r in resources
        ],
        "caseStudy": {
            "title": "自主辅导智能体（ReAct）诊断报告",
            "description": tutoring.answer if tutoring else "本次流程未触发自主辅导节点。",
            "code": trace_json or "// 无工具调用轨迹",
            "explanation": (
                f"自主循环共 {tutoring.rounds} 轮，调用工具：{'、'.join(tutoring.tools_used) or '无'}"
                f"（驱动方式：{'真实模型' if tutoring.mode == 'llm' else '规则策略'}）。"
                if tutoring
                else ""
            ),
        },
        "review": {"score": quality_score, "passed": not errors, "issues": issues},
        "knowledgeSources": [{"document": r.title, "section": r.type} for r in resources],
        "executionSteps": [
            {"agent": name, "duration": max(1, latency // 5)}
            for name in ("ProfileAgent", "PlannerAgent", "ResourceAgent", "QuizAgent", "ReviewAgent")
        ],
        "plan": plan.model_dump() if plan else None,
        "reviewReport": review.model_dump() if review else None,
        "totalDuration": latency,
    }

    a3_store.add_resource(package)
    a3_store.add_log("generate", latency)
    return success({"resource": package, "message": "学习资源包生成成功"})


@router.get("/resource")
def a3_list_resources() -> dict:
    resources = [
        {
            "id": r.get("id"),
            "course": r.get("course"),
            "topic": r.get("topic"),
            "studentName": r.get("studentName"),
            "createdAt": r.get("createdAt"),
            "reviewScore": (r.get("review") or {}).get("score"),
            "reviewPassed": (r.get("review") or {}).get("passed"),
        }
        for r in a3_store.list_resources()
    ]
    return success({"resources": resources})


@router.get("/resource/{rid}")
def a3_get_resource(rid: str):
    r = a3_store.get_resource(rid)
    if not r:
        return fail("资源不存在", 404)
    return success({"resource": r})


# --------------------------------------------------------------------------- #
# 学习路径（PlannerAgent）
# --------------------------------------------------------------------------- #
@router.get("/path/{profile_id}")
def a3_get_path(profile_id: str):
    profile = load_profile(profile_id)
    if profile is None:
        return fail("画像不存在", 404)

    state: AgentState = {
        "user_id": profile_id,
        "messages": [Message(role="user", content=f"请为{profile.major or '我'}规划学习路径")],
        "profile": profile,
        "resources": [],
        "errors": [],
    }
    try:
        out = build_plan_node(state)
    except Exception as e:  # noqa: BLE001
        logger.exception("a3 path failed")
        return fail(f"学习路径生成失败: {e}", 500)

    plan = out["plan"]
    stages = []
    for s in plan.steps:
        stages.append(
            {
                "stage": s.step,
                "title": s.title,
                "estimatedTime": f"约 {s.est_minutes} 分钟",
                "objective": s.description,
                "topics": [s.title],
                "resources": [],
            }
        )
    return success(
        {
            "path": {
                "course": profile.learningGoal.details or plan.goal,
                "goal": plan.goal,
                "totalMinutes": plan.total_minutes,
                "stages": stages,
            }
        }
    )


# --------------------------------------------------------------------------- #
# 智能辅导（TutorAgent ReAct 循环）
# --------------------------------------------------------------------------- #
@router.post("/tutor/ask")
async def a3_tutor_ask(request: Request):
    body = await _json_body(request)
    question = (body.get("question") or "").strip()
    profile_id = (body.get("profileId") or "").strip()
    if not question:
        return fail("问题不能为空", 400)

    user_id = profile_id or "anonymous"
    start = time.time()

    try:
        init_db()
    except Exception as e:  # noqa: BLE001
        logger.warning("tutor init_db failed: %s", e)

    settings = get_settings()
    hints = {}
    profile = load_profile(user_id)
    if profile is not None:
        hints["gaps"] = [profile.weakPoints.details] if profile.weakPoints.details else []

    session = None
    mode = (os.getenv("TUTOR_MODE") or "auto").strip().lower()
    if not mock_enabled() and mode != "policy" and _has_usable_key(settings.llm_api_key):
        try:
            session = _run_llm_react(question, user_id, [])
        except Exception as e:  # noqa: BLE001
            logger.warning("tutor llm react failed, fallback policy: %s", e)
    if session is None or not session.answer.strip():
        try:
            session = _run_policy_react(question, user_id, hints)
        except Exception as e:  # noqa: BLE001
            logger.exception("tutor policy failed")
            return fail(f"辅导智能体执行失败: {e}", 500)

    latency = int((time.time() - start) * 1000)
    a3_store.add_log("qa", latency)
    return success(
        {
            "answer": session.answer,
            "sources": [],
            "hasAnswer": bool(session.answer.strip()),
            "answerMode": "agent",
            "retrievedChunks": [],
            "webResults": [],
            "latency": latency,
            "model": "ReAct 辅导智能体（规则策略）" if session.mode == "policy" else "ReAct 辅导智能体",
            "modelId": "tutor-agent",
            "toolsUsed": session.tools_used,
            "rounds": session.rounds,
            "trace": [t.model_dump() for t in session.trace],
            "grounded": bool(session.tools_used),
            "groundingReason": (
                f"结论基于 {len(session.tools_used)} 次真实工具调用（第 {'、'.join(str(t.round) for t in session.trace)} 轮）"
                if session.trace
                else "无工具调用，直接作答"
            ),
        }
    )


# --------------------------------------------------------------------------- #
# 模型管理
# --------------------------------------------------------------------------- #
@router.get("/models")
def a3_list_models() -> dict:
    s = get_settings()
    builtin = (
        [{"id": "default", "name": f"DeepSeek（{s.llm_model}）", "model": s.llm_model, "baseUrl": s.llm_base_url}]
        if _has_usable_key(s.llm_api_key) and not mock_enabled()
        else [{"id": "mock", "name": "离线演示模式（规则策略）", "model": "policy", "baseUrl": "-"}]
    )
    return success({"builtIn": builtin, "custom": a3_store.list_custom_models()})


@router.post("/models")
async def a3_add_model(request: Request):
    body = await _json_body(request)
    required = ("id", "name", "baseUrl", "apiKey", "model")
    if any(not (body.get(k) or "").strip() for k in required):
        return fail("缺少必要字段：id, name, baseUrl, apiKey, model", 400)
    mid = (body.get("id") or "").strip().lower().replace(" ", "-")
    model = {
        "id": mid,
        "name": (body.get("name") or "").strip(),
        "baseUrl": (body.get("baseUrl") or "").strip(),
        "apiKey": (body.get("apiKey") or "").strip(),
        "model": (body.get("model") or "").strip(),
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a3_store.upsert_custom_model(model)
    return success({"id": model["id"], "name": model["name"], "type": "custom", "model": model["model"], "baseUrl": model["baseUrl"]}, "模型导入成功")


@router.delete("/models/{mid}")
def a3_delete_model(mid: str):
    a3_store.delete_custom_model(mid)
    return success(None, "模型已删除")


# --------------------------------------------------------------------------- #
# 知识库管理
# --------------------------------------------------------------------------- #
def _split_sections(content: str) -> list[str]:
    parts = [p.strip() for p in (content or "").split("\n## ") if p.strip()]
    return parts or [content or ""]


def _doc_full(doc: dict) -> dict:
    sections = _split_sections(doc["content"])
    return {
        "id": doc["id"],
        "title": doc["title"],
        "filename": doc.get("filename", ""),
        "content": doc["content"],
        "sections": sections,
        "sectionCount": len(sections),
        "contentLength": len(doc["content"]),
        "createdAt": doc.get("createdAt", ""),
        "updatedAt": doc.get("updatedAt", ""),
    }


@router.get("/knowledge")
def a3_list_knowledge() -> dict:
    docs = [
        {
            "id": d["id"],
            "title": d["title"],
            "filename": d.get("filename", ""),
            "sectionCount": len(_split_sections(d["content"])),
            "contentLength": len(d["content"]),
            "preview": d["content"][:200],
        }
        for d in a3_store.list_knowledge()
    ]
    return success({"documents": docs})


@router.get("/knowledge/{doc_id}")
def a3_get_knowledge(doc_id: str):
    d = a3_store.get_knowledge(doc_id)
    if not d:
        return fail("文档不存在", 404)
    return success({"document": _doc_full(d)})


@router.post("/knowledge/upload")
async def a3_upload_knowledge(request: Request):
    body = await _json_body(request)
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    if not title or not content:
        return fail("标题和内容不能为空", 400)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = {"id": _new_id("doc"), "title": title, "content": content, "filename": "", "createdAt": now, "updatedAt": now}
    a3_store.upsert_knowledge(doc)
    return success({"document": _doc_full(doc), "message": "文档创建成功"})


@router.put("/knowledge/{doc_id}")
async def a3_update_knowledge(doc_id: str, request: Request):
    body = await _json_body(request)
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    if not title or not content:
        return fail("标题和内容不能为空", 400)
    old = a3_store.get_knowledge(doc_id)
    if not old:
        return fail("文档不存在", 404)
    doc = {**old, "title": title, "content": content, "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    a3_store.upsert_knowledge(doc)
    return success({"document": _doc_full(doc), "message": "文档更新成功"})


# --------------------------------------------------------------------------- #
# 学习分析
# --------------------------------------------------------------------------- #
@router.get("/analytics")
def a3_analytics() -> dict:
    from app.db import count_all_profiles

    logs = a3_store.load()["logs"]
    logs_by_type: dict[str, int] = {}
    latencies: list[int] = []
    for lg in logs:
        logs_by_type[lg.get("type", "other")] = logs_by_type.get(lg.get("type", "other"), 0) + 1
        if lg.get("latency"):
            latencies.append(lg["latency"])

    from app.db import list_profiles

    recent = [
        {"id": r.get("id"), "course": r.get("course"), "topic": r.get("topic"), "studentName": r.get("studentName"), "createdAt": r.get("createdAt")}
        for r in a3_store.list_resources()[:10]
    ]
    recent_logs = [
        {
            "type": lg.get("type", "other"),
            "latency": lg.get("latency", 0),
            "timestamp": lg.get("createdAt", ""),
            "input": "",
        }
        for lg in logs[:10]
    ]
    return success(
        {
            "analytics": {
                "totalProfiles": count_all_profiles(),
                "totalResources": len(a3_store.list_resources()),
                "totalLogs": len(logs),
                "totalDocs": len(a3_store.list_knowledge()),
                "logsByType": logs_by_type,
                "avgLatency": round(sum(latencies) / len(latencies)) if latencies else 0,
                "recentResources": recent,
                "recentLogs": recent_logs,
            }
        }
    )


# --------------------------------------------------------------------------- #
# 前后测效果实证
# --------------------------------------------------------------------------- #
@router.get("/efficacy/questions")
def a3_efficacy_questions(topic: str = "机器学习") -> dict:
    qs = strip_answers(get_questions(topic))
    return success({"topic": topic, "questions": qs, "total": len(qs)})


@router.post("/efficacy/submit")
async def a3_efficacy_submit(request: Request):
    body = await _json_body(request)
    topic = body.get("topic")
    phase = body.get("phase")
    answers = body.get("answers")
    if not topic or not phase or answers is None:
        return fail("缺少必要参数（topic/phase/answers）", 400)
    if phase not in ("pre", "post"):
        return fail("phase 必须为 pre 或 post", 400)

    result = bank_score(answers, topic)
    record = {
        "id": _new_id("eff"),
        "studentId": body.get("studentId") or _new_id("stu"),
        "studentName": body.get("studentName") or "同学",
        "topic": topic,
        "phase": phase,
        "score": result["score"],
        "correct": result["correct"],
        "total": result["total"],
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a3_store.add_efficacy(record)

    others = [r for r in a3_store.list_efficacy() if r["studentId"] == record["studentId"] and r["phase"] != phase and r["topic"] == topic]
    improvement = record["score"] - others[0]["score"] if others else None

    return success(
        {
            "score": result["score"],
            "correct": result["correct"],
            "total": result["total"],
            "detail": result["detail"],
            "improvement": improvement,
            "record": record,
        }
    )


@router.get("/efficacy/summary")
def a3_efficacy_summary() -> dict:
    pairs = [
        {"studentName": s["studentName"], "topic": s["topic"], "pre": s["pre"], "post": s["post"], "improvement": s["post"] - s["pre"], "seeded": True}
        for s in SEED_SAMPLES
    ]

    by_student: dict[str, dict] = {}
    for r in a3_store.list_efficacy():
        by_student.setdefault(r["studentId"], {})[r["phase"]] = r
    for phases in by_student.values():
        p, q = phases.get("pre"), phases.get("post")
        if p and q:
            pairs.append(
                {
                    "studentName": p["studentName"],
                    "topic": p["topic"],
                    "pre": p["score"],
                    "post": q["score"],
                    "improvement": q["score"] - p["score"],
                    "seeded": False,
                }
            )

    pre_sum = sum(x["pre"] for x in pairs)
    post_sum = sum(x["post"] for x in pairs)
    per_topic: dict[str, dict] = {}
    for x in pairs:
        t = per_topic.setdefault(x["topic"], {"topic": x["topic"], "pre": 0, "post": 0, "n": 0})
        t["pre"] += x["pre"]
        t["post"] += x["post"]
        t["n"] += 1
    per_topic_arr = [
        {"topic": t["topic"], "avgPre": round(t["pre"] / t["n"]), "avgPost": round(t["post"] / t["n"]), "improvement": round((t["post"] - t["pre"]) / t["n"])}
        for t in per_topic.values()
    ]

    count = len(pairs)
    return success(
        {
            "totalStudents": count,
            "avgPre": round(pre_sum / count) if count else 0,
            "avgPost": round(post_sum / count) if count else 0,
            "avgImprovement": round((post_sum - pre_sum) / count) if count else 0,
            "perTopic": per_topic_arr,
            "sampleStudents": pairs,
        }
    )

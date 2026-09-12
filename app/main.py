"""FastAPI 入口。

端点：
- GET  /health                 健康检查
- POST /api/profile/build      仅构建学生画像
- POST /api/learn              跑完整多智能体流程（画像→计划→资源→测验→复盘）
- GET  /api/profile/{user_id}  从数据库读取已保存画像
- POST /api/auth/register      昵称+密码注册
- POST /api/auth/login         昵称+密码登录
- POST /api/auth/phone/login   手机号+密码登录
- POST /api/auth/third/login   微信/QQ 扫码登录（首次需手机号或邮箱验证码验证）
- POST /api/auth/bind/contact  已登录用户绑定手机号/邮箱
- GET  /api/auth/me            当前登录用户信息
- GET  /api/auth/me/identities 已绑定的账号身份列表
- POST /api/verify/send        发送验证码（腾讯云短信 / 邮箱 SMTP）
- POST /api/verify/check       校验验证码
- GET  /api/verify/channels    通道状态（真实下发 or 演示模式）
- GET  /api/metrics            运行指标（?format=prometheus 为 Prometheus 文本格式）
- GET  /api/traces             最近若干次请求的 trace_id
- GET  /api/traces/{trace_id}  单次请求的链路明细（各节点耗时 / 计数器 / 报错）

运行：
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from app import auth as auth_mod
from app import telemetry
from app.agents.profile_agent import build_profile_node
from app.db import get_engine, init_db, load_profile, save_profile
from app.graph import build_graph
from app.models import (
    AgentState,
    LearnRequest,
    LearnResponse,
    Message,
    Profile,
    ProfileBuildRequest,
)
from app.verify import channel_status, detect_channel, send_verify_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learning-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Python Learning Agent", version="0.1.0", lifespan=lifespan)
graph = build_graph()

# A3 前端兼容层必须先于下方 /api/profile/{user_id} 注册：
# 两者对 GET /api/profile/{id} 的响应结构不同（信封包装 vs 裸 Profile），
# FastAPI 按注册顺序匹配，先注册者生效。
from app.a3_compat import router as a3_router  # noqa: E402

app.include_router(a3_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "python-learning-agent"}


@app.post("/api/profile/build", response_model=Profile)
def api_build_profile(req: ProfileBuildRequest, response: Response) -> Profile:
    telemetry.bump(telemetry.C_REQUESTS)
    with telemetry.trace_run() as trace:
        state: AgentState = {
            "user_id": req.user_id,
            "messages": req.messages,
            "resources": [],
            "errors": [],
        }
        try:
            out = build_profile_node(state)
        except Exception as e:  # noqa: BLE001
            logger.exception("profile build failed")
            raise HTTPException(status_code=500, detail=f"画像生成失败: {e}")

        profile = out["profile"]
        try:
            save_profile(req.user_id, profile)
        except Exception as e:  # noqa: BLE001
            logger.warning("save profile failed: %s", e)
    response.headers["X-Trace-Id"] = trace.trace_id
    return profile


@app.post("/api/learn", response_model=LearnResponse)
def api_learn(req: LearnRequest, response: Response) -> LearnResponse:
    telemetry.bump(telemetry.C_REQUESTS)
    # 整条管线包在一条 trace 里：节点级 span、模型调用数、防幻觉拦截次数
    # 都会归到这次请求名下，出错也照样落盘（便于事后查「哪一步炸的」）。
    with telemetry.trace_run() as trace:
        out = _run_learn(req)
    response.headers["X-Trace-Id"] = trace.trace_id
    return out


def _run_learn(req: LearnRequest) -> LearnResponse:
    state: AgentState = {
        "user_id": req.user_id,
        "messages": req.messages,
        "resources": [],
        "errors": [],
    }
    try:
        final = graph.invoke(state)
    except Exception as e:  # noqa: BLE001
        logger.exception("learn pipeline failed")
        raise HTTPException(status_code=500, detail=f"学习流程执行失败: {e}")

    profile = final.get("profile")
    if profile is None:
        raise HTTPException(status_code=500, detail="画像生成为空")

    try:
        save_profile(req.user_id, profile)
    except Exception as e:  # noqa: BLE001
        logger.warning("save profile failed: %s", e)

    return LearnResponse(
        user_id=req.user_id,
        profile=profile,
        plan=final.get("plan"),
        resources=final.get("resources", []),
        quiz=final.get("quiz"),
        review=final.get("review"),
        tutoring=final.get("tutoring"),
        errors=final.get("errors", []),
    )


# ============================ 可观测性 ============================
@app.get("/api/metrics")
def api_metrics(format: str = "json"):
    """运行指标：JSON（默认）或 Prometheus 文本格式。

    `guardrails` 一节专门回答「防幻觉到底生效没有」——把三道代码级约束的
    拦截动作变成了计数器：检索空命中、代码级拒答、被白名单拦下的编造链接。
    """
    if format == "prometheus":
        return PlainTextResponse(telemetry.render_prometheus())
    return {"ok": True, "metrics": telemetry.snapshot()}


@app.get("/api/traces")
def api_traces(limit: int = 10) -> dict:
    """最近若干次请求的 trace_id（新的在前）。"""
    return {"ok": True, "trace_ids": telemetry.latest_trace_ids(max(1, min(limit, 50)))}


@app.get("/api/traces/{trace_id}")
def api_trace(trace_id: str) -> dict:
    """单次请求的链路明细：走了哪些节点、各耗时多少、哪些计数器被触发。"""
    trace = telemetry.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="未找到该 trace（仅保留最近若干条）")
    return {"ok": True, "trace": trace}


@app.get("/api/profile/{user_id}", response_model=Profile)
def api_get_profile(user_id: str) -> Profile:
    profile = load_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="未找到该用户画像")
    return profile


# ============================ 账号 / 验证码 ============================
def get_session():
    engine = get_engine()
    s = DBSession(engine)
    try:
        yield s
    finally:
        s.close()


def get_current_user(
    session: DBSession = Depends(get_session),
    authorization: str = Header(default=""),
) -> auth_mod.User:
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    payload = auth_mod.verify_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="请先登录或登录已过期")
    user = session.get(auth_mod.User, payload["uid"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


class AuthIn(BaseModel):
    nickname: str = ""
    password: str = ""
    phone: str = ""


class VerifySendIn(BaseModel):
    target: str = ""
    scene: str = "bind"


class VerifyCheckIn(BaseModel):
    target: str = ""
    code: str = ""
    scene: str = "bind"


class ThirdLoginIn(BaseModel):
    provider: str = ""
    code: str = ""
    target: str | None = None
    verifyCode: str | None = None


class BindContactIn(BaseModel):
    target: str = ""
    verifyCode: str = ""


@app.post("/api/auth/register")
def api_register(body: AuthIn, session: DBSession = Depends(get_session)) -> dict:
    try:
        return auth_mod.register(session, body.nickname, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
def api_login(body: AuthIn, session: DBSession = Depends(get_session)) -> dict:
    try:
        return auth_mod.login(session, body.nickname, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/phone/login")
def api_phone_login(body: AuthIn, session: DBSession = Depends(get_session)) -> dict:
    try:
        return auth_mod.phone_login(session, body.phone, body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/verify/send")
def api_verify_send(body: VerifySendIn, session: DBSession = Depends(get_session)) -> dict:
    channel = detect_channel(body.target)
    if not channel:
        raise HTTPException(status_code=400, detail="请输入正确的手机号或邮箱")
    scene = "third_login" if body.scene == "third_login" else "bind"
    code = auth_mod.create_verify_code(session, body.target, channel, scene)
    r = send_verify_code(body.target, channel, code)
    return {
        "ok": True,
        "channel": channel,
        "scene": scene,
        "delivered": r.get("delivered", False),
        "devCode": r.get("devCode"),
        "expireSec": 300,
    }


@app.get("/api/verify/channels")
def api_verify_channels() -> dict:
    return {"ok": True, "channels": channel_status()}


@app.post("/api/verify/check")
def api_verify_check(body: VerifyCheckIn, session: DBSession = Depends(get_session)) -> dict:
    scene = "third_login" if body.scene == "third_login" else "bind"
    ok, err = auth_mod.check_verify_code(session, body.target, body.code, scene)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True}


@app.post("/api/auth/third/login")
async def api_third_login(body: ThirdLoginIn, session: DBSession = Depends(get_session)) -> dict:
    r = await auth_mod.oauth_login(session, body.provider, body.code, body.target, body.verifyCode)
    if not r.get("ok") and not r.get("needBind"):
        raise HTTPException(status_code=401, detail=r.get("error", "登录失败"))
    return r


@app.post("/api/auth/bind/contact")
def api_bind_contact(
    body: BindContactIn,
    user: auth_mod.User = Depends(get_current_user),
    session: DBSession = Depends(get_session),
) -> dict:
    r = auth_mod.bind_contact(session, user.id, body.target, body.verifyCode)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "绑定失败"))
    return r


@app.get("/api/auth/me")
def api_me(user: auth_mod.User = Depends(get_current_user)) -> dict:
    return {"ok": True, "user": {"id": user.id, "nickname": user.nickname, "avatar": user.avatar, "role": user.role}}


@app.get("/api/auth/me/identities")
def api_my_identities(
    user: auth_mod.User = Depends(get_current_user),
    session: DBSession = Depends(get_session),
) -> dict:
    return {"ok": True, "identities": auth_mod.get_identities(session, user.id)}

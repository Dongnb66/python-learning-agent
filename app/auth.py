"""账号与鉴权：用户体系、JWT、手机号/邮箱验证码、微信/QQ 绑定。

设计要点（与 travel-rank / campus-mutual-aid 保持一致的工程口径）：
- 密码用标准库 hashlib.scrypt 加盐哈希，不引入 passlib / bcrypt 依赖。
- JWT（HS256）用 hmac + base64url 手工实现，不引入 PyJWT 依赖。
- 验证码只存哈希不存明文；同一 target+scene 的旧码自动作废，防止并存绕过。
- 微信/QQ 首次登录必须先验证手机号或邮箱，避免产生无法找回的「僵尸账号」。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime

from sqlalchemy import Integer, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db import Base  # 复用同一 metadata，init_db() 时自动建表

ACCESS_TTL_SEC = 900  # access 15 分钟
VERIFY_TTL_MS = 5 * 60 * 1000  # 验证码 5 分钟

# 未配置 JWT_SECRET 时用随机密钥，避免硬编码默认值被伪造（重启后旧 token 失效）
# 注意：这里**不在模块顶层做 os.getenv 快照** —— 模块级读取会固化成
# 「导入这一刻的环境变量」，一旦导入发生在 .env 注入之前，.env 里配的
# JWT_SECRET 就永远生效不了（重启后登录态照样失效，且没有任何提示）。
_JWT_SECRET_FALLBACK = secrets.token_hex(32)


def _jwt_secret() -> str:
    """JWT 签名密钥（调用时读取，未配置则使用进程级随机密钥）。"""
    return (os.getenv("JWT_SECRET") or "").strip() or _JWT_SECRET_FALLBACK

AVATARS = ["🎓", "📚", "🧠", "🚀", "🌱", "🔍", "💡", "🛠️"]


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ------------------------------ ORM 模型 ------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nickname: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    avatar: Mapped[str] = mapped_column(String(16), default="🎓")
    role: Mapped[str] = mapped_column(String(16), default="student")
    created_at: Mapped[str] = mapped_column(String(32), default=_now_str)


class UserIdentity(Base):
    """同一用户可绑定 手机号 / 邮箱 / 微信 / QQ。"""

    __tablename__ = "user_identities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    provider: Mapped[str] = mapped_column(String(16))  # phone / email / wechat / qq
    external_id: Mapped[str] = mapped_column(String(64))
    verified: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String(32), default=_now_str)


class VerifyCode(Base):
    __tablename__ = "verify_codes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    target: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(16))
    code_hash: Mapped[str] = mapped_column(String(64))
    scene: Mapped[str] = mapped_column(String(32))
    expired_at: Mapped[int] = mapped_column(Integer)
    used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(32), default=_now_str)


# ------------------------------ 密码 ------------------------------
def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.scrypt(pw.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1, dklen=64).hex()
    return f"{salt}:{h}"


def verify_password(pw: str, stored: str) -> bool:
    if not stored or ":" not in stored:
        return False
    salt, h = stored.split(":", 1)
    calc = hashlib.scrypt(pw.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1, dklen=64).hex()
    return hmac.compare_digest(calc, h)


# ------------------------------ JWT ------------------------------
def sign_jwt(payload: dict, ttl_sec: int = ACCESS_TTL_SEC) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + ttl_sec
    head = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body_b64 = _b64url(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode())
    signing_input = f"{head}.{body_b64}".encode()
    sig = hmac.new(_jwt_secret().encode(), signing_input, hashlib.sha256).digest()
    return f"{head}.{body_b64}.{_b64url(sig)}"


def verify_jwt(token: str) -> dict | None:
    try:
        head, body_b64, sig = token.split(".")
        expected = _b64url(hmac.new(_jwt_secret().encode(), f"{head}.{body_b64}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig):
            return None
        payload = json.loads(_b64url_decode(body_b64))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return payload
    except Exception:
        return None


def issue_tokens(user: User) -> dict:
    payload = {"uid": user.id, "nickname": user.nickname, "role": user.role}
    return {
        "accessToken": sign_jwt(payload),
        "expiresIn": ACCESS_TTL_SEC,
        "user": {"id": user.id, "nickname": user.nickname, "avatar": user.avatar, "role": user.role},
    }


# ------------------------------ 验证码 ------------------------------
def create_verify_code(session: Session, target: str, channel: str, scene: str, ttl_ms: int = VERIFY_TTL_MS) -> str:
    code = f"{secrets.randbelow(900000) + 100000}"
    code_hash = hashlib.sha256(f"{target}|{code}".encode()).hexdigest()
    # 同 target+scene 旧码全部作废
    for row in session.scalars(select(VerifyCode).where(VerifyCode.target == target, VerifyCode.scene == scene, VerifyCode.used == 0)):
        row.used = 1
    session.add(
        VerifyCode(
            target=target,
            channel=channel,
            code_hash=code_hash,
            scene=scene,
            expired_at=int(time.time() * 1000) + ttl_ms,
        )
    )
    session.commit()
    return code


def check_verify_code(session: Session, target: str, code: str, scene: str) -> tuple[bool, str]:
    row = (
        session.scalars(
            select(VerifyCode)
            .where(VerifyCode.target == target, VerifyCode.scene == scene)
            .order_by(VerifyCode.id.desc())
        ).first()
    )
    if not row:
        return False, "请先获取验证码"
    if row.used:
        return False, "验证码已使用，请重新获取"
    if row.expired_at < int(time.time() * 1000):
        return False, "验证码已过期，请重新获取"
    if not hmac.compare_digest(row.code_hash, hashlib.sha256(f"{target}|{code}".encode()).hexdigest()):
        return False, "验证码不正确"
    row.used = 1
    session.commit()
    return True, ""


# ------------------------------ 身份绑定 ------------------------------
def find_identity(session: Session, provider: str, external_id: str) -> UserIdentity | None:
    return session.scalars(
        select(UserIdentity).where(UserIdentity.provider == provider, UserIdentity.external_id == external_id)
    ).first()


def bind_identity(session: Session, user_id: int, provider: str, external_id: str) -> None:
    if find_identity(session, provider, external_id):
        return
    session.add(UserIdentity(user_id=user_id, provider=provider, external_id=external_id, verified=1))
    session.commit()


def get_identities(session: Session, user_id: int) -> list[dict]:
    rows = session.scalars(select(UserIdentity).where(UserIdentity.user_id == user_id)).all()
    return [{"provider": r.provider, "externalId": r.external_id, "verified": r.verified} for r in rows]


# ------------------------------ 业务：注册 / 登录 ------------------------------
def register(session: Session, nickname: str, password: str) -> dict:
    nickname = (nickname or "").strip()
    if not nickname or not password:
        raise ValueError("昵称和密码不能为空")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    if session.scalars(select(User).where(User.nickname == nickname)).first():
        raise ValueError("该昵称已被注册")
    user = User(
        nickname=nickname,
        password_hash=hash_password(password),
        avatar=secrets.choice(AVATARS),
        role="student",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return issue_tokens(user)


def login(session: Session, nickname: str, password: str) -> dict:
    user = session.scalars(select(User).where(User.nickname == (nickname or "").strip())).first()
    if not user or not verify_password(password or "", user.password_hash):
        raise ValueError("昵称或密码错误")
    return issue_tokens(user)


def phone_login(session: Session, phone: str, password: str) -> dict:
    ident = find_identity(session, "phone", phone)
    if not ident:
        raise ValueError("该手机号未注册")
    user = session.get(User, ident.user_id)
    if not user or not verify_password(password or "", user.password_hash):
        raise ValueError("手机号或密码错误")
    return issue_tokens(user)


async def _code_to_openid(provider: str, code: str) -> str | None:
    """真实环境需配置 appid/secret 换取 openid；未配置时降级为 mock（code 当 openid）。"""
    import urllib.request

    appid = os.getenv("WECHAT_APPID") if provider == "wechat" else os.getenv("QQ_APPID")
    secret = os.getenv("WECHAT_SECRET") if provider == "wechat" else os.getenv("QQ_SECRET")
    if not appid or not secret or not code:
        return code
    try:
        if provider == "wechat":
            url = (
                f"https://api.weixin.qq.com/sns/oauth2/access_token?appid={appid}"
                f"&secret={secret}&code={code}&grant_type=authorization_code"
            )
        else:
            url = (
                f"https://graph.qq.com/oauth2.0/token?grant_type=authorization_code"
                f"&client_id={appid}&client_secret={secret}&code={code}"
                f"&redirect_uri={os.getenv('QQ_REDIRECT', '')}"
            )
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("openid")
    except Exception:
        return code


async def oauth_login(session: Session, provider: str, code: str, target: str | None, verify_code: str | None) -> dict:
    """微信/QQ 扫码登录：已绑定直接进；首次需手机号或邮箱 + 验证码。"""
    if provider not in ("wechat", "qq"):
        return {"ok": False, "error": "不支持的第三方"}
    openid = await _code_to_openid(provider, code)
    if not openid:
        return {"ok": False, "needBind": True, "error": "授权失败，请重新扫码"}

    ident = find_identity(session, provider, openid)
    if ident:
        user = session.get(User, ident.user_id)
        return {"ok": True, **issue_tokens(user)}

    if not target or not verify_code:
        return {
            "ok": False,
            "needBind": True,
            "provider": provider,
            "openid": openid,
            "error": "首次使用微信/QQ 登录，请先绑定手机号或邮箱完成验证",
        }

    from app.verify import detect_channel

    channel = detect_channel(target)
    if not channel:
        return {"ok": False, "needBind": True, "error": "请输入正确的手机号或邮箱"}
    ok, err = check_verify_code(session, target, verify_code, "third_login")
    if not ok:
        return {"ok": False, "needBind": True, "error": err}

    exist = find_identity(session, channel, target)
    if exist:
        user = session.get(User, exist.user_id)
    else:
        user = User(
            nickname=f"{'微信' if provider == 'wechat' else 'QQ'}用户{openid[:4]}",
            password_hash=hash_password(secrets.token_hex(12)),
            avatar=secrets.choice(AVATARS),
            role="student",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    bind_identity(session, user.id, channel, target)
    bind_identity(session, user.id, provider, openid)
    return {"ok": True, **issue_tokens(user)}


def bind_contact(session: Session, user_id: int, target: str, verify_code: str) -> dict:
    """已登录用户绑定手机号/邮箱（需验证码）。"""
    from app.verify import detect_channel

    channel = detect_channel(target)
    if not channel:
        return {"ok": False, "error": "请输入正确的手机号或邮箱"}
    ok, err = check_verify_code(session, target, verify_code, "bind")
    if not ok:
        return {"ok": False, "error": err}
    if find_identity(session, channel, target):
        return {"ok": False, "error": "该邮箱已被绑定" if channel == "email" else "该手机号已被绑定"}
    bind_identity(session, user_id, channel, target)
    return {"ok": True, "channel": channel}

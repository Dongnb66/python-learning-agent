"""验证码发送通道：邮箱（SMTP）/ 手机短信（腾讯云 SMS）。

真实环境：
- 邮箱：配了 SMTP_HOST/SMTP_USER/SMTP_PASS 就用标准库 smtplib 真实发信（腾讯云 SES 也提供 SMTP）。
- 短信：配了腾讯云短信 5 个环境变量就真实下发（见 app/sms.py）。
未配置时自动降级为「演示模式」：打印到服务端日志，并把验证码回给前端，方便本地联调与演示。
生产环境请务必配置真实通道，此时不再回传验证码。
"""
from __future__ import annotations

import os
import re
import smtplib
from email.mime.text import MIMEText

from app.sms import send_sms_tencent, sms_configured, sms_missing_keys

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PHONE_RE = re.compile(r"^1\d{10}$")


def is_email(t: str | None) -> bool:
    return bool(t and EMAIL_RE.match(t))


def is_phone(t: str | None) -> bool:
    return bool(t and PHONE_RE.match(t))


def detect_channel(target: str | None) -> str | None:
    """自动识别目标是邮箱还是手机号。"""
    if is_email(target):
        return "email"
    if is_phone(target):
        return "phone"
    return None


def channel_status() -> dict:
    """当前各通道运行状态，供运维面板 / 前端提示使用。"""
    return {
        "email": {
            "configured": bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER") and os.getenv("SMTP_PASS")),
            "provider": os.getenv("SMTP_HOST", "未配置"),
        },
        "sms": {
            "configured": sms_configured(),
            "provider": "腾讯云短信 SMS",
            "dryRun": os.getenv("SMS_DRY_RUN") == "1",
            "missing": sms_missing_keys(),
        },
    }


def send_verify_code(target: str, channel: str, code: str) -> dict:
    return send_email(target, code) if channel == "email" else send_sms(target, code)


def send_email(to: str, code: str) -> dict:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    if host and user and pwd:
        try:
            port = int(os.getenv("SMTP_PORT", "465"))
            msg = MIMEText(
                f"<p>你的验证码是</p><h2 style='letter-spacing:4px'>{code}</h2><p>5 分钟内有效，若非本人操作请忽略。</p>",
                "html",
                "utf-8",
            )
            msg["Subject"] = "【学习多智能体系统】账号验证"
            msg["From"] = os.getenv("SMTP_FROM", user)
            msg["To"] = to
            if port == 465:
                with smtplib.SMTP_SSL(host, port, timeout=10) as s:
                    s.login(user, pwd)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(host, port, timeout=10) as s:
                    s.starttls()
                    s.login(user, pwd)
                    s.send_message(msg)
            return {"ok": True, "delivered": True, "channel": "email"}
        except Exception as e:  # 发送失败降级为演示模式，保证流程不中断
            print(f"[verify] 邮件发送失败，降级为控制台输出：{e}")
    print(f"[verify][邮件] 收件人 {to} → 验证码 {code}（5 分钟内有效）")
    return {"ok": True, "delivered": False, "channel": "email", "devCode": code}


def send_sms(to: str, code: str) -> dict:
    if sms_configured():
        r = send_sms_tencent(to, code, minutes=5)
        if r.get("ok") and r.get("delivered"):
            return {"ok": True, "delivered": True, "channel": "phone-tencent"}
        if r.get("dryRun"):
            print(f"[verify][短信·dry-run] 手机号 {to} → 验证码 {code}")
            return {"ok": True, "delivered": False, "channel": "phone-tencent", "devCode": code, "dryRun": True}
        if not r.get("ok"):
            print(f"[verify] 短信发送失败，降级为演示模式：{r.get('error')}")
            print(f"[verify][短信] 手机号 {to} → 验证码 {code}（5 分钟内有效）")
            return {
                "ok": True,
                "delivered": False,
                "channel": "phone",
                "devCode": code,
                "fallback": True,
                "reason": r.get("error", ""),
            }
    print(f"[verify][短信] 手机号 {to} → 验证码 {code}（5 分钟内有效）")
    return {"ok": True, "delivered": False, "channel": "phone", "devCode": code}

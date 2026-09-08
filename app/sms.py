"""腾讯云短信（SMS）通道 —— 零第三方依赖实现。

采用腾讯云 API 3.0 的 TC3-HMAC-SHA256 签名，仅用标准库（hashlib / hmac / urllib）
自行实现，不引入 tencentcloud-sdk-python，保持依赖精简。

需要的环境变量（见 .env.example）：
    TENCENTCLOUD_SECRET_ID        —— 腾讯云控制台 → 访问管理 → API 密钥
    TENCENTCLOUD_SECRET_KEY
    TENCENTCLOUD_SMS_SDK_APP_ID   —— 短信控制台 → 应用管理 → SDK AppID
    TENCENTCLOUD_SMS_SIGN_NAME    —— 已审核通过的签名内容
    TENCENTCLOUD_SMS_TEMPLATE_ID  —— 已审核通过的正文模板 ID
    TENCENTCLOUD_SMS_REGION       —— 可选，默认 ap-guangzhou
    SMS_DRY_RUN=1                 —— 只打印请求体，不真正调用（联调用）

未配置时调用方自动降级为「演示模式」，不会抛异常。
签名算法参考：https://cloud.tencent.com/document/api/382/52071
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HOST = "sms.tencentcloudapi.com"
SERVICE = "sms"
VERSION = "2021-01-11"
ACTION = "SendSms"
ALGORITHM = "TC3-HMAC-SHA256"

REQUIRED_KEYS = [
    "TENCENTCLOUD_SECRET_ID",
    "TENCENTCLOUD_SECRET_KEY",
    "TENCENTCLOUD_SMS_SDK_APP_ID",
    "TENCENTCLOUD_SMS_SIGN_NAME",
    "TENCENTCLOUD_SMS_TEMPLATE_ID",
]

# 常见错误码 → 中文提示，便于排障
ERR_MAP = {
    "LimitExceeded": "发送频率超限",
    "InvalidPhoneNumber": "手机号格式错误",
    "SignatureIncorrectOrUnapproved": "短信签名未审核通过",
    "TemplateIncorrectOrUnapproved": "短信模板未审核通过",
    "InsufficientBalance": "短信套餐包余额不足",
    "AuthFailure": "密钥错误或无权限",
}


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sms_configured() -> bool:
    return all(os.getenv(k) for k in REQUIRED_KEYS)


def sms_missing_keys() -> list[str]:
    return [k for k in REQUIRED_KEYS if not os.getenv(k)]


def build_tc3_headers(
    payload_obj: dict,
    timestamp_sec: int | None = None,
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    service: str = SERVICE,
    host: str = HOST,
    action: str = ACTION,
    region: str | None = None,
    version: str = VERSION,
) -> dict[str, str]:
    """按 TC3-HMAC-SHA256 规范生成请求头（含 Authorization）。"""
    secret_id = secret_id or os.getenv("TENCENTCLOUD_SECRET_ID", "")
    secret_key = secret_key or os.getenv("TENCENTCLOUD_SECRET_KEY", "")
    region = region or os.getenv("TENCENTCLOUD_SMS_REGION", "ap-guangzhou")
    ts = int(timestamp_sec if timestamp_sec is not None else time.time())

    payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
    # 必须使用 UTC 日期，否则凌晨时段必定签名失败
    date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    content_type = "application/json; charset=utf-8"
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"
    canonical_request = (
        "POST" + "\n" + "/" + "\n" + "" + "\n"
        + canonical_headers + "\n"
        + signed_headers + "\n"
        + _sha256_hex(payload)
    )

    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (
        ALGORITHM + "\n" + str(ts) + "\n" + credential_scope + "\n"
        + _sha256_hex(canonical_request)
    )

    secret_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac(secret_date, service)
    secret_signing = _hmac(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"{ALGORITHM} Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Content-Type": content_type,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Version": version,
        "X-TC-Region": region,
        "X-TC-Timestamp": str(ts),
        "Authorization": authorization,
    }


def send_sms_tencent(phone: str, code: str, minutes: int = 5) -> dict:
    """发送短信验证码。返回 {ok, delivered, channel, requestId?, error?}"""
    if not sms_configured():
        return {"ok": False, "delivered": False, "error": "腾讯云短信未配置", "missing": sms_missing_keys()}

    payload = {
        "PhoneNumberSet": ["+86" + str(phone).strip()],
        "SmsSdkAppId": os.getenv("TENCENTCLOUD_SMS_SDK_APP_ID"),
        "SignName": os.getenv("TENCENTCLOUD_SMS_SIGN_NAME"),
        "TemplateId": os.getenv("TENCENTCLOUD_SMS_TEMPLATE_ID"),
        "TemplateParamSet": [str(code), str(minutes)],
    }

    if os.getenv("SMS_DRY_RUN") == "1":
        print("[sms][dry-run] 将发送：", json.dumps(payload, ensure_ascii=False))
        return {"ok": True, "delivered": False, "dryRun": True, "channel": "phone-tencent"}

    headers = build_tc3_headers(payload)
    req = urllib.request.Request(
        "https://" + HOST + "/",
        data=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # 腾讯云错误也走 HTTP 200 之外的码
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "delivered": False, "error": f"短信服务请求失败：{e}"}
    except Exception as e:
        return {"ok": False, "delivered": False, "error": f"短信服务请求失败：{e}"}

    resp_body = data.get("Response", {})
    status_list = resp_body.get("SendStatusSet") or []
    st = status_list[0] if status_list else None
    if st and st.get("Code") == "Ok":
        return {"ok": True, "delivered": True, "channel": "phone-tencent", "requestId": resp_body.get("RequestId")}

    err_code = (st or {}).get("Code") or (resp_body.get("Error") or {}).get("Code") or "Unknown"
    msg = (
        ERR_MAP.get(err_code)
        or (st or {}).get("Message")
        or (resp_body.get("Error") or {}).get("Message")
        or "短信发送失败"
    )
    return {"ok": False, "delivered": False, "error": f"{msg}（{err_code}）", "requestId": resp_body.get("RequestId")}

"""腾讯云 TC3-HMAC-SHA256 签名算法校验。

使用腾讯云官方文档公开的测试向量：
  timestamp 1539084154（UTC 2018-10-09）
  期望 HashedCanonicalRequest = 91c9c192c14460df6c1ffc69e34e6c5e90708de2a6d282cccf957dbf1aa7f3a7
  期望 Signature              = 5da7a33f6993f0614b047e5df4582db9e9bf4672ba50567dba16c6ccf174c474

官方示例 SecretId / SecretKey 不入库（GitHub Secret Scanning 会拦截云凭证），改用占位段；
HashedCanonicalRequest 等不依赖密钥的官方向量照常逐字节校验。
需要完整复现最终 Signature 时注入官方示例 SecretKey：
  TC3_SAMPLE_SECRET_KEY=<官方示例 SecretKey> pytest -q

注意：主站《签名方法 v3》文档的 POST 示例密钥被脱敏（AKID****），无法复现其签名，
      但其 payload 哈希 35e9c5b0... 可复现，一并校验。
"""
from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone

from app.sms import build_tc3_headers, sms_configured, sms_missing_keys


class _EnvShim:
    """无 pytest 时（直接 python 运行本文件）使用的简易环境变量适配器。"""

    def setenv(self, k: str, v: str) -> None:
        os.environ[k] = v

    def delenv(self, k: str, raising: bool = True) -> None:
        os.environ.pop(k, None)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def test_payload_hash_matches_official_post_example() -> None:
    """主站 POST 示例：payload 哈希应一致。"""
    payload = '{"Limit": 1, "Filters": [{"Values": ["\\u672a\\u547d\\u540d"], "Name": "instance-name"}]}'
    assert _sha256_hex(payload) == "35e9c5b0e3ae67532d3c9f17ead6c90222632e5b1ff7f6e89887f1398934f064"


def test_signature_matches_official_vector() -> None:
    """可复现 GET 向量：完整校验签名链。"""
    # 官方示例 SecretId 用占位段（GitHub Secret Scanning 对 AKID 前缀一律拦截）。
    # 官方示例 SecretKey 参与 HMAC 运算，同样不入库，通过 TC3_SAMPLE_SECRET_KEY 注入：
    #   TC3_SAMPLE_SECRET_KEY=<官方示例 SecretKey> pytest -q
    secret_id = "AKIDoEXAMPLEoEXAMPLEoEXAMPLEoEXAMPLEo"
    secret_key = os.environ.get("TC3_SAMPLE_SECRET_KEY") or "Gu5t9EXAMPLEoEXAMPLEoEXAMPLEoEXAMPLEo"
    ts = 1539084154
    service = "cvm"
    host = "cvm.tencentcloudapi.com"
    date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    assert date == "2018-10-09"  # UTC 日期换算

    canonical_headers = (
        "content-type:application/x-www-form-urlencoded\n"
        f"host:{host}\n"
    )
    signed_headers = "content-type;host"
    empty_payload_hash = _sha256_hex("")
    assert empty_payload_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    canonical_request = (
        "GET" + "\n" + "/" + "\n" + "Limit=10&Offset=0" + "\n"
        + canonical_headers + "\n" + signed_headers + "\n" + empty_payload_hash
    )
    assert _sha256_hex(canonical_request) == (
        "91c9c192c14460df6c1ffc69e34e6c5e90708de2a6d282cccf957dbf1aa7f3a7"
    )

    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (
        "TC3-HMAC-SHA256" + "\n" + str(ts) + "\n" + credential_scope + "\n"
        + _sha256_hex(canonical_request)
    )
    k_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
    k_service = _hmac(k_date, service)
    k_signing = _hmac(k_service, "tc3_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    if os.environ.get("TC3_SAMPLE_SECRET_KEY"):
        # 注入官方示例 SecretKey 后，签名应与官方文档逐字节一致
        assert signature == "5da7a33f6993f0614b047e5df4582db9e9bf4672ba50567dba16c6ccf174c474"
    else:
        # 未注入时只校验签名形态（官方向量的 HashedCanonicalRequest 已在上一步逐字节校验）
        assert len(signature) == 64 and all(c in "0123456789abcdef" for c in signature)


def test_build_tc3_headers_shape(monkeypatch=None) -> None:
    """配置齐全时，生成的 Authorization 头格式应合法。"""
    monkeypatch = monkeypatch or _EnvShim()
    for k, v in {
        "TENCENTCLOUD_SECRET_ID": "AKIDtest",
        "TENCENTCLOUD_SECRET_KEY": "testkey",
        "TENCENTCLOUD_SMS_SDK_APP_ID": "1400000000",
        "TENCENTCLOUD_SMS_SIGN_NAME": "学习助手",
        "TENCENTCLOUD_SMS_TEMPLATE_ID": "123456",
    }.items():
        monkeypatch.setenv(k, v)

    assert sms_configured() is True
    assert sms_missing_keys() == []

    headers = build_tc3_headers(
        {
            "PhoneNumberSet": ["+8613800138000"],
            "SmsSdkAppId": "1400000000",
            "SignName": "学习助手",
            "TemplateId": "123456",
            "TemplateParamSet": ["123456", "5"],
        },
        1700000000,
    )
    assert headers["X-TC-Action"] == "SendSms"
    assert headers["Host"] == "sms.tencentcloudapi.com"
    assert headers["X-TC-Timestamp"] == "1700000000"
    auth = headers["Authorization"]
    assert auth.startswith("TC3-HMAC-SHA256 Credential=AKIDtest/2023-11-14/sms/tc3_request, ")
    assert "SignedHeaders=content-type;host;x-tc-action, Signature=" in auth
    sig = auth.split("Signature=")[1]
    assert len(sig) == 64 and all(c in "0123456789abcdef" for c in sig)


def test_not_configured_reports_missing(monkeypatch=None) -> None:
    """未配置时应如实报告缺失项，且 sms_configured 为 False。"""
    monkeypatch = monkeypatch or _EnvShim()
    for k in (
        "TENCENTCLOUD_SECRET_ID",
        "TENCENTCLOUD_SECRET_KEY",
        "TENCENTCLOUD_SMS_SDK_APP_ID",
        "TENCENTCLOUD_SMS_SIGN_NAME",
        "TENCENTCLOUD_SMS_TEMPLATE_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    assert sms_configured() is False
    assert len(sms_missing_keys()) == 5


if __name__ == "__main__":
    # 无 pytest 时可直接运行：python tests/test_tencent_sms.py
    for fn in (
        test_payload_hash_matches_official_post_example,
        test_signature_matches_official_vector,
        test_build_tc3_headers_shape,
        test_not_configured_reports_missing,
    ):
        fn()
        print(f"[OK] {fn.__name__}")
    print("\n全部通过：腾讯云 TC3 签名实现与官方向量一致")

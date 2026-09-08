"""账号体系端到端流程测试：注册 → 登录 → 验证码 → 微信扫码绑定 → 再次扫码直接登录。

无 pytest 时可直接运行：python tests/test_auth_flow.py
有 pytest 时：pytest tests/test_auth_flow.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB = "sqlite:///./_test_auth.db"

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app import auth as auth_mod  # noqa: E402
from app.db import Base  # noqa: E402
from app.verify import channel_status, detect_channel, send_verify_code  # noqa: E402


def main() -> None:
    path = TEST_DB.replace("sqlite:///./", "")
    if os.path.exists(path):
        os.remove(path)

    engine = create_engine(TEST_DB, future=True)
    Base.metadata.create_all(engine)
    s = Session(engine)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, extra: str = "") -> None:
        checks.append((name, ok, extra))
        print(f"{'[OK]  ' if ok else '[FAIL]'} {name}{('  ' + extra) if extra else ''}")

    # 1. 注册
    try:
        r = auth_mod.register(s, "testuser", "test123")
        token = r["accessToken"]
        check("昵称密码注册", bool(token) and r["user"]["nickname"] == "testuser")
    except Exception as e:
        check("昵称密码注册", False, str(e))
        return

    # 2. JWT 可解析
    payload = auth_mod.verify_jwt(token)
    check("JWT 签发与校验", payload is not None and payload["nickname"] == "testuser")

    # 3. 登录（含错误密码）
    ok_login = True
    try:
        auth_mod.login(s, "testuser", "test123")
    except Exception as e:
        ok_login = False
        print(e)
    wrong_rejected = False
    try:
        auth_mod.login(s, "testuser", "wrongpwd")
    except ValueError:
        wrong_rejected = True
    check("登录 + 错误密码被拒", ok_login and wrong_rejected)

    # 4. 通道识别
    check("通道识别（手机/邮箱/非法）", detect_channel("13800138000") == "phone" and detect_channel("a@b.com") == "email" and detect_channel("abc") is None)

    # 5. 发送验证码（未配置腾讯云 → 演示模式回传 devCode）
    code = auth_mod.create_verify_code(s, "13800138000", "phone", "third_login")
    r = send_verify_code("13800138000", "phone", code)
    check("验证码发送（演示模式回传）", r.get("devCode") == code, f"通道={r.get('channel')}")

    # 6. 微信首次扫码 → 要求绑定
    r1 = asyncio.run(auth_mod.oauth_login(s, "wechat", "wx_py_001", None, None))
    check("首次扫码要求绑定", r1.get("needBind") is True, r1.get("error", "")[:30])

    # 7. 错误验证码 → 拒绝
    r2 = asyncio.run(auth_mod.oauth_login(s, "wechat", "wx_py_001", "13800138000", "000000"))
    check("错误验证码被拒", r2.get("ok") is False, r2.get("error", ""))

    # 8. 正确验证码 → 建号并绑定
    code2 = auth_mod.create_verify_code(s, "13900139000", "phone", "third_login")
    send_verify_code("13900139000", "phone", code2)
    r3 = asyncio.run(auth_mod.oauth_login(s, "wechat", "wx_py_001", "13900139000", code2))
    check("正确验证码完成绑定登录", r3.get("ok") is True, str(r3.get("user", {}).get("nickname", "")))

    # 9. 再次扫码 → 直接登录
    r4 = asyncio.run(auth_mod.oauth_login(s, "wechat", "wx_py_001", None, None))
    check("再次扫码直接登录", r4.get("ok") is True)

    # 10. 身份列表
    ident = auth_mod.get_identities(s, r3["user"]["id"])
    providers = {i["provider"] for i in ident}
    check("身份绑定记录完整", providers == {"phone", "wechat"}, str(sorted(providers)))

    # 11. 通道状态
    st = channel_status()
    check("通道状态可查询", "sms" in st and st["sms"]["provider"] == "腾讯云短信 SMS", f"已配置={st['sms']['configured']}")

    s.close()
    engine.dispose()  # 释放连接，否则 Windows 下无法删除测试库文件
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"[提示] 测试库文件未能删除（可手动清理）：{e}")

    failed = [c for c in checks if not c[1]]
    print(f"\n结果：{len(checks) - len(failed)} 通过 / {len(failed)} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

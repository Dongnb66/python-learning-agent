"""环境变量加载测试 —— 锁住「.env 里的配置其实读不到」这个真 bug。

背景
----
本项目用 pydantic-settings 读 `.env`，但它**只把值读进 Settings 对象，
不会写进 os.environ**。而项目里 JWT 密钥、腾讯云短信、SMTP、微信/QQ
开放平台这些配置都是直接用 `os.getenv` 读的。

于是在修复前，`cp .env.example .env` 并把 Key / Secret 填好之后：
- `app.py`（一键 Gradio 演示）判 `USE_MOCK = not bool(os.getenv("LLM_API_KEY"))`
  → 永远 True，Demo 一直跑 mock 假数据；
- 短信永远走「演示模式」、微信/QQ 永远 mock 扫码、JWT_SECRET 每次启动随机生成
  （重启后登录态全失效）；
- 而且全过程**没有任何报错或提示**，看代码还以为配置生效了。

本文件把这些行为固化成断言。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app import config


# ------------------------- load_env_file 本身 -------------------------
def test_load_env_file_injects_into_os_environ(tmp_path: Path) -> None:
    """核心行为：.env 的内容必须真的进 os.environ（这是修复的关键）。"""
    env_file = tmp_path / ".env"
    env_file.write_text('ENV_TEST_KEY="from-file"\nENV_TEST_OTHER=plain\n', encoding="utf-8")
    os.environ.pop("ENV_TEST_KEY", None)
    os.environ.pop("ENV_TEST_OTHER", None)

    assert config.load_env_file(env_file) is True
    assert os.getenv("ENV_TEST_KEY") == "from-file"
    assert os.getenv("ENV_TEST_OTHER") == "plain"

    os.environ.pop("ENV_TEST_KEY", None)
    os.environ.pop("ENV_TEST_OTHER", None)


def test_load_env_file_does_not_override_shell_env(tmp_path: Path) -> None:
    """部署时应能用 shell 变量覆盖 .env，所以默认不覆盖已有变量。"""
    env_file = tmp_path / ".env"
    env_file.write_text("ENV_TEST_KEEP=from-file\n", encoding="utf-8")
    os.environ["ENV_TEST_KEEP"] = "from-shell"

    config.load_env_file(env_file)

    assert os.getenv("ENV_TEST_KEEP") == "from-shell"

    os.environ.pop("ENV_TEST_KEEP", None)


def test_load_env_file_missing_file_returns_false(tmp_path: Path) -> None:
    """没有 .env 时不应抛异常（CI / 容器里常见）。"""
    assert config.load_env_file(tmp_path / "not-exists.env") is False


def test_default_env_file_points_to_project_root() -> None:
    """.env 必须锚定项目根，而不是「当前工作目录」——否则换个目录启动就读不到。"""
    assert config.DEFAULT_ENV_FILE == config.PROJECT_ROOT / ".env"
    assert (config.PROJECT_ROOT / "requirements.txt").exists()


def test_importing_config_already_loaded_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """导入 config 就应该完成注入（不需要调用方记得手动 load）。"""
    env_file = tmp_path / ".env"
    env_file.write_text("ENV_TEST_IMPORT=yes\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", env_file)
    os.environ.pop("ENV_TEST_IMPORT", None)

    # 模拟「模块被首次导入」时的副作用
    assert config.load_env_file() is True
    assert os.getenv("ENV_TEST_IMPORT") == "yes"

    os.environ.pop("ENV_TEST_IMPORT", None)


def test_settings_env_file_is_absolute_not_cwd_relative() -> None:
    """pydantic-settings 的 env_file 也必须是绝对路径，与 load_env_file 保持一致。"""
    env_file = config.Settings.model_config["env_file"]
    assert Path(env_file).is_absolute()
    assert Path(env_file).name == ".env"


# ------------------------- 占位符 / 可用性判定 -------------------------
@pytest.mark.parametrize(
    "key,expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("sk-your-api-key-here", False),   # .env.example 里的占位符
        ("sk-xxxxxxxxxxxx", False),
        ("change-me", False),
        ("sk-abcdef1234567890", True),
        ("sk-1a2b3c4d5e6f", True),
    ],
)
def test_has_usable_llm_key(key: str | None, expected: bool) -> None:
    """占位符 Key 视为「未配置」，避免每次都白等一轮注定 401 的请求。"""
    assert config.has_usable_llm_key(key) is expected


def test_llm_configured_matches_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm_configured() 必须跟 Settings 里读到的值一致（也就是认识 .env）。"""
    settings = config.get_settings()
    monkeypatch.setattr(settings, "llm_api_key", "sk-real-key-123")
    assert config.llm_configured() is True

    monkeypatch.setattr(settings, "llm_api_key", "sk-your-api-key-here")
    assert config.llm_configured() is False

    monkeypatch.setattr(settings, "llm_api_key", "")
    assert config.llm_configured() is False


def test_placeholder_rules_are_shared_with_tutor_agent() -> None:
    """占位符规则只应有一份：tutor_agent 复用 config 的实现，不另起一套。"""
    from app.agents import tutor_agent

    assert tutor_agent._PLACEHOLDER_MARKERS is config._PLACEHOLDER_MARKERS
    for key in ("sk-your-api-key-here", "", "sk-real123"):
        assert tutor_agent._has_usable_key(key) == config.has_usable_llm_key(key)


# ------------------------- JWT_SECRET 的读时取值 -------------------------
def test_jwt_secret_reads_env_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """JWT_SECRET 必须在调用时读取。

    修复前是模块级 `JWT_SECRET = os.getenv(...) or 随机值` ——
    导入那一刻没拿到 .env 的值，就会永久固化成随机密钥，
    表现为「.env 里配了 JWT_SECRET，重启后登录态照样全失效」。
    """
    from app import auth

    monkeypatch.setenv("JWT_SECRET", "configured-secret")
    assert auth._jwt_secret() == "configured-secret"

    monkeypatch.delenv("JWT_SECRET", raising=False)
    fallback = auth._jwt_secret()
    assert fallback  # 未配置时回退到随机密钥，不抛异常
    assert auth._jwt_secret() == fallback  # 同一进程内稳定，否则 token 无法校验


def test_jwt_sign_verify_roundtrip_uses_current_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """签名与校验都走同一个密钥，避免改成读时取值后两边不一致。"""
    from app import auth

    monkeypatch.setenv("JWT_SECRET", "roundtrip-secret")
    token = auth.sign_jwt({"uid": 1, "nickname": "tester"})
    payload = auth.verify_jwt(token)
    assert payload is not None
    assert payload["uid"] == 1

    # 换一个密钥后，旧 token 必须校验失败
    monkeypatch.setenv("JWT_SECRET", "another-secret")
    assert auth.verify_jwt(token) is None

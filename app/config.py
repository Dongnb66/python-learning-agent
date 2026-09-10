"""配置层：从环境变量 / .env 读取 LLM 与数据库相关配置。

设计要点：
- 兼容 DeepSeek / OpenAI / Claude 等 OpenAI 协议兼容端点。
- 默认 base_url 指向 DeepSeek，模型默认 deepseek-chat（便宜、支持 function calling）。
- 通过 pydantic-settings 实现类型安全的配置读取。
- **同时把 .env 注入 os.environ**：见下方 load_env_file() 的说明。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级）。.env 固定在项目根，
# 避免「换一个目录启动就读不到配置」。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# 明显是占位符的 Key，视为「未配置」，避免每次白跑一次注定 401 的请求
_PLACEHOLDER_MARKERS = ("your", "xxx", "placeholder", "change", "example", "todo", "none")


def has_usable_llm_key(api_key: str | None) -> bool:
    """判断是否配置了「看起来可用」的 LLM Key。

    本地 / Demo 环境常见 `sk-your-api-key-here` 这类占位符，
    若当成真 Key 去调用，每轮都会白等一次网络超时再降级——不如提前识别。
    """
    key = (api_key or "").strip()
    if not key:
        return False
    low = key.lower()
    return not any(marker in low for marker in _PLACEHOLDER_MARKERS)


def load_env_file(
    path: str | os.PathLike | None = None, *, override: bool = False
) -> bool:
    """把 .env 注入 os.environ，返回是否真的加载到了文件。

    为什么必须显式做这一步：
    pydantic-settings 只把 .env 读进自己的 Settings 对象，**不会**写进
    os.environ。而本项目里 JWT 密钥、腾讯云短信、SMTP、微信/QQ 开放平台
    这些配置都是直接用 os.getenv 读的 —— 不注入的话，写在 .env 里等于没配：
    验证码永远走「演示模式」、第三方登录永远 mock 扫码、JWT_SECRET 每次
    启动随机生成（重启后登录态全失效），而且不会有任何报错提示。

    已存在的 shell 环境变量默认优先（override=False），方便部署时覆盖 .env。
    """
    target = Path(path) if path is not None else DEFAULT_ENV_FILE
    if not target.exists():
        return False
    return bool(load_dotenv(target, override=override))


# 导入即注入：config 是所有入口（app.py / app/main.py / scripts）都会经过的一层
load_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM（OpenAI 协议兼容）----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ---- 数据库 ----
    database_url: str = "sqlite:///./learning_agent.db"

    # ---- 应用 ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"


_settings: Settings | None = None


def get_settings() -> Settings:
    """返回进程级单例配置。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def llm_configured() -> bool:
    """当前是否配置了可用的 LLM Key（占位符视为未配置）。

    入口判断真实/Mock 模式时用它，而不是 `bool(os.getenv("LLM_API_KEY"))`：
    后者既不看 .env（除非已注入），也不认识占位符。
    """
    return has_usable_llm_key(get_settings().llm_api_key)

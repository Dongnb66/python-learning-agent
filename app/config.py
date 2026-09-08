"""配置层：从环境变量 / .env 读取 LLM 与数据库相关配置。

设计要点：
- 兼容 DeepSeek / OpenAI / Claude 等 OpenAI 协议兼容端点。
- 默认 base_url 指向 DeepSeek，模型默认 deepseek-chat（便宜、支持 function calling）。
- 通过 pydantic-settings 实现类型安全的配置读取。
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
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

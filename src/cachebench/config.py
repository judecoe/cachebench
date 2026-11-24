"""Gateway settings, read from CACHEBENCH_* environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHEBENCH_")

    redis_url: str = "redis://localhost:6379/0"
    backend: Literal["fake", "openai_compat"] = "fake"
    local_base_url: str = "http://localhost:11434/v1"

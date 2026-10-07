import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from src.config.paths import PROJECT_ROOT


ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_name: str
    log_level: str
    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    stock_api_provider: str | None
    stock_api_key: str | None
    stock_api_secret: str | None
    stock_api_url: str | None
    news_api_provider: str | None
    news_api_key: str | None
    vector_db_provider: str
    vector_db_path: str


def load_settings(env_path: Path = ENV_PATH) -> Settings:
    env_values = dotenv_values(env_path) if env_path.exists() else {}

    def get_value(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name, env_values.get(name))
        if value is None or not value.strip():
            return default
        return value.strip()

    return Settings(
        app_env=get_value("APP_ENV", "development") or "development",
        app_name=get_value("APP_NAME", "RAG Stock Advisor") or "RAG Stock Advisor",
        log_level=get_value("LOG_LEVEL", "INFO") or "INFO",
        openai_api_key=get_value("OPENAI_API_KEY"),
        openai_model=get_value("OPENAI_MODEL", "gpt-4o") or "gpt-4o",
        openai_embedding_model=(
            get_value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            or "text-embedding-3-small"
        ),
        stock_api_provider=get_value("STOCK_API_PROVIDER"),
        stock_api_key=get_value("STOCK_API_KEY"),
        stock_api_secret=get_value("STOCK_API_SECRET"),
        stock_api_url=get_value("STOCK_API_URL"),
        news_api_provider=get_value("NEWS_API_PROVIDER"),
        news_api_key=get_value("NEWS_API_KEY"),
        vector_db_provider=get_value("VECTOR_DB_PROVIDER", "faiss") or "faiss",
        vector_db_path=(
            get_value("VECTOR_DB_PATH", "data/vector_store/faiss")
            or "data/vector_store/faiss"
        ),
    )


settings = load_settings()

import os
import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from src.config.paths import PROJECT_ROOT


ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True, kw_only=True)
class Settings:
    app_env: str
    app_name: str
    log_level: str
    openai_api_key: str | None
    openai_reasoning: str = "medium"
    openai_model: str
    openai_embedding_model: str
    stock_api_provider: str | None
    stock_api_key: str | None
    stock_api_secret: str | None
    stock_api_url: str | None
    news_api_provider: str | None
    news_api_key: str | None
    news_api_secret: str | None = None
    vector_db_provider: str
    vector_db_path: str
    # 질문 전체 결정, 인덱싱 예산과 종목별 추가 조회 예산
    agent_max_decisions: int = 3
    agent_max_news_pages: int = 2
    agent_max_stock_refreshes: int = 1
    agent_max_indexing_calls: int = 2
    agent_max_index_chunks: int = 1000  # 인덱싱 1회 처리량 1,000청크
    agent_max_index_characters: int = 500000  # 인덱싱 1회 처리량 50만 문자
    agent_timeout_seconds: float = 120.0  # 질문 분석 후 초기 조회, Agent 반복의 경과 시간 상한
    agent_request_timeout_seconds: float = 20.0 # 개별 요청별 타임아웃
    agent_news_refresh_seconds: float = 900.0  # 동일 종목·기간·페이지 재수집 대기


def load_settings(env_path: Path = ENV_PATH) -> Settings:
    env_values = dotenv_values(env_path) if env_path.exists() else {}

    def get_value(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name, env_values.get(name))
        if value is None or not value.strip():
            return default
        return value.strip()

    # 비용·시간 제한은 양수만 허용하며 잘못된 환경변수는 시작 시 거부
    def get_limit(name: str, default: int | float) -> int | float:
        value = type(default)(get_value(name, str(default)))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
        return value

    return Settings(
        app_env=get_value("APP_ENV", "development") or "development",
        app_name=get_value("APP_NAME", "RAG Stock Advisor") or "RAG Stock Advisor",
        log_level=get_value("LOG_LEVEL", "INFO") or "INFO",
        openai_api_key=get_value("OPENAI_API_KEY"),
        openai_reasoning=get_value("OPENAI_REASONING", "medium") or "medium",
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
        news_api_secret=get_value("NEWS_API_SECRET"),
        vector_db_provider=get_value("VECTOR_DB_PROVIDER", "faiss") or "faiss",
        vector_db_path=(
            get_value("VECTOR_DB_PATH", "data/vector_store/faiss")
            or "data/vector_store/faiss"
        ),
        agent_max_decisions=get_limit("AGENT_MAX_DECISIONS", 3),
        agent_max_news_pages=min(2, get_limit("AGENT_MAX_NEWS_PAGES", 2)),
        agent_max_stock_refreshes=get_limit("AGENT_MAX_STOCK_REFRESHES", 1),
        agent_max_indexing_calls=get_limit("AGENT_MAX_INDEXING_CALLS", 2),
        agent_max_index_chunks=get_limit("AGENT_MAX_INDEX_CHUNKS", 1000),
        agent_max_index_characters=get_limit("AGENT_MAX_INDEX_CHARACTERS", 500000),
        agent_timeout_seconds=get_limit("AGENT_TIMEOUT_SECONDS", 120.0),
        agent_request_timeout_seconds=get_limit("AGENT_REQUEST_TIMEOUT_SECONDS", 20.0),
        agent_news_refresh_seconds=get_limit("AGENT_NEWS_REFRESH_SECONDS", 900.0),
    )


settings = load_settings()

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.config.paths import SAMPLE_NEWS_PATH


NewsRecord = dict[str, Any]


# 전체 json을 가져옴
def load_sample_news(file_path: Path = SAMPLE_NEWS_PATH) -> list[NewsRecord]:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("sample_news.json must contain a JSON array.")

    return data


# 종목명 기준으로 뉴스 데이터 필터링
def load_sample_news_by_name(stock_name: str) -> list[NewsRecord]:
    news_items = load_sample_news()
    matched_news = [
        news for news in news_items if news.get("stock_name") == stock_name
    ]

    if not matched_news:
        raise ValueError(f"Sample news data not found: {stock_name}")

    return matched_news


# 종목명 기준으로 최신 뉴스부터 가져옴
def load_recent_sample_news(
    stock_name: str,
    limit: int | None = None,
) -> list[NewsRecord]:
    matched_news = load_sample_news_by_name(stock_name)
    sorted_news = sorted(
        matched_news,
        key=lambda news: date.fromisoformat(news["published_date"]),
        reverse=True,
    )

    if limit is None:
        return sorted_news

    return sorted_news[:limit]

from typing import Any
from dataclasses import asdict
from src.domain.stock_models import StockTrend
from src.parsers.stock_question_parser import ParsedQuestion

StockRecord = dict[str, Any]
NewsRecord = dict[str, Any]
StockContext = dict[str, Any]

DEFAULT_SENTIMENTS = ("positive", "neutral", "negative")


# 질문 분석 결과, 주가 데이터, 뉴스 데이터 => 하나의 분석 컨텍스트로 구성
def build_stock_context(
    parsed_question: ParsedQuestion,
    stock_data: StockRecord,
    news_data: list[NewsRecord],
    trend_data: StockTrend | None = None,
) -> StockContext:
    return {
        "question": {
            "original_question": parsed_question.original_question,
            "intent": parsed_question.intent,
            "recent_days": parsed_question.recent_days,
        },
        "stock": {
            **stock_data,
            "trend": trend_data.trend if trend_data is not None else stock_data["trend"],
        },
        "stock_trend": asdict(trend_data) if trend_data is not None else None,
        "data_sources": {"stock": stock_data.get("source", "sample"), "news": "sample"},
        "news": _build_news_context(news_data),
        "summary": _build_summary(news_data),
    }


# 응답 생성에 필요한 뉴스 필드만 추림
def _build_news_context(news_data: list[NewsRecord]) -> list[NewsRecord]:
    return [
        {
            "title": news["title"],
            "summary": news["summary"],
            "sentiment": news["sentiment"],
            "published_date": news["published_date"],
        }
        for news in news_data
    ]


# 뉴스 감성 개수를 요약
def _build_summary(news_data: list[NewsRecord]) -> dict[str, int]:
    summary = {f"{label}_news_count": 0 for label in DEFAULT_SENTIMENTS} # 모두 0으로 초기값 세팅

    for news in news_data:
        sentiment = news.get("sentiment", "neutral") # 해당 값이 없으면 neutral로 처리
        key = f"{sentiment}_news_count"

        if key not in summary:
            summary["neutral_news_count"] += 1
            continue

        summary[key] += 1

    return summary

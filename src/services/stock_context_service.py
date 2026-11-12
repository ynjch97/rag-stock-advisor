from typing import Any
from dataclasses import asdict
from src.domain.stock_models import StockTrend
from src.domain.advice_models import ParsedQuestion

StockRecord = dict[str, Any]
NewsRecord = dict[str, Any]
StockContext = dict[str, Any]

# 질문 분석 결과, 주가 데이터, 뉴스 데이터 => 하나의 분석 컨텍스트로 구성
def build_stock_context(
    parsed_question: ParsedQuestion,
    stock_data: StockRecord,
    news_data: list[NewsRecord],
    trend_data: StockTrend | None = None,
    *, news_source: str = "sample",
) -> StockContext:
    news = _build_news_context(news_data)
    article_keys = {
        (item.get("stock_code"), item.get("url") or item.get("news_id")
         or (item["title"], item["published_date"])) for item in news
    }
    return {
        "question": {
            "original_question": parsed_question.original_question,
            "intent": parsed_question.intent,
            "recent_days": parsed_question.recent_days,
            "stocks": [asdict(stock) for stock in parsed_question.stocks],
            "period": asdict(parsed_question.period),
        },
        "stock": {
            **stock_data,
            "trend": trend_data.trend if trend_data is not None else stock_data["trend"],
        },
        "stock_trend": asdict(trend_data) if trend_data is not None else None,
        "data_sources": {"stock": stock_data.get("source", "sample"), "news": news_source},
        "news": news,
        "summary": {"news_count": len(article_keys), "news_chunk_count": len(news)},
    }


# 응답 생성에 필요한 뉴스 필드만 추림
def _build_news_context(news_data: list[NewsRecord]) -> list[NewsRecord]:
    return [
        {
            "title": news["title"],
            "content": news.get("content") or news.get("summary", ""), # 샘플 요약과 검색 본문을 content로 통일
            "published_date": news["published_date"],
            **{key: news[key] for key in (
                "news_id", "chunk_id", "chunk_index", "chunk_count", "start_index",
                "stock_name", "stock_code", "url", "publisher", "important",
                "vector_score", "keyword_score", "vector_normalized_score",
                "keyword_normalized_score", "hybrid_score",
            ) if key in news},
        }
        for news in news_data
    ]

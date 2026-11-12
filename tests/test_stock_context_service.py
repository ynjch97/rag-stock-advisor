from copy import deepcopy

from src.domain.advice_models import ParsedQuestion, ParsedStock
from src.services.stock_context_service import build_stock_context


def test_build_stock_context_combines_question_stock_and_news():
    context = build_stock_context(
        parsed_question=ParsedQuestion(
            original_question="삼성전자 사도 될까?",
            stocks=(ParsedStock("삼성전자", "삼성전자", "005930"),),
            intent="buy_opinion",
        ),
        stock_data={
            "stock_name": "삼성전자",
            "stock_code": "005930",
            "current_price": 78600,
            "change_rate": 0.5,
            "trend": "최근 5거래일 약세 후 반등 시도",
            "base_date": "2026-07-08",
        },
        news_data=[
            {
                "title": "긍정 뉴스",
                "summary": "긍정 요약",
                "published_date": "2026-07-07",
            },
            {
                "title": "부정 뉴스",
                "summary": "부정 요약",
                "published_date": "2026-07-07",
            },
        ],
    )

    assert context["question"]["intent"] == "buy_opinion"
    assert context["stock"]["stock_code"] == "005930"
    assert len(context["news"]) == 2
    assert context["summary"] == {"news_count": 2, "news_chunk_count": 2}
    assert context["news"][0]["content"] == "긍정 요약"
    assert all("sentiment" not in news for news in context["news"])


# 같은 기사의 여러 청크는 보존하되 기사 수는 중복 집계하지 않음
def test_hybrid_news_preserves_content_sources_scores_and_article_count():
    article = {
        "news_id": "news-1", "stock_code": "005930", "stock_name": "삼성전자",
        "title": "반도체 실적", "published_date": "2026-11-01", "publisher": "경제신문",
        "url": "https://example.test/article", "important": "O", "chunk_count": 2,
        "vector_score": 0.7, "keyword_score": None, "vector_normalized_score": 0.8,
        "keyword_normalized_score": 0, "hybrid_score": 0.56,
    }
    news = [dict(article, chunk_id="news-1_001", chunk_index=1, content="첫 본문", start_index=0),
            dict(article, chunk_id="news-1_002", chunk_index=2, content="뒤 본문", start_index=10)]
    before = deepcopy(news)
    context = build_stock_context(
        ParsedQuestion("삼성전자 전망", (ParsedStock("삼성전자", "삼성전자", "005930"),), "general_analysis"),
        {"stock_code": "005930", "source": "toss", "trend": "상승"}, news, news_source="hybrid_search",
    )
    assert context["news"] == news
    assert context["summary"] == {"news_count": 1, "news_chunk_count": 2}
    assert context["data_sources"] == {"stock": "toss", "news": "hybrid_search"}
    assert news == before


# 빈 실제 검색 결과를 샘플로 잘못 표시하지 않음
def test_empty_hybrid_news_context():
    context = build_stock_context(
        ParsedQuestion("질문", (ParsedStock("삼성전자", "삼성전자", "005930"),), "general_analysis"),
        {"source": "toss", "trend": "보합"}, [], news_source="hybrid_search",
    )
    assert context["news"] == []
    assert context["summary"] == {"news_count": 0, "news_chunk_count": 0}
    assert context["data_sources"]["news"] == "hybrid_search"

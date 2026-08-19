from src.services.stock_context_service import build_stock_context


def test_build_stock_context_combines_question_stock_and_news():
    context = build_stock_context(
        parsed_question={
            "original_question": "삼성전자 사도 될까?",
            "stock_name": "삼성전자",
            "intent": "buy_opinion",
        },
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
                "sentiment": "positive",
                "published_date": "2026-07-07",
            },
            {
                "title": "부정 뉴스",
                "summary": "부정 요약",
                "sentiment": "negative",
                "published_date": "2026-07-07",
            },
        ],
    )

    assert context["question"]["intent"] == "buy_opinion"
    assert context["stock"]["stock_code"] == "005930"
    assert len(context["news"]) == 2
    assert context["summary"] == {
        "positive_news_count": 1,
        "neutral_news_count": 0,
        "negative_news_count": 1,
    }

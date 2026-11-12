from copy import deepcopy

import pytest

from src.services.stock_advice_service import generate_stock_advice, generate_stock_comparison_advice


# 주가·뉴스를 응답 템플릿에 넣을 최소 컨텍스트 제공
def _context(news=None, intent="general_analysis", change_rate=1):
    return {
        "question": {"intent": intent},
        "stock": {"stock_name": "삼성전자", "current_price": 100, "base_date": "2026-11-01",
                  "trend": "최근 5거래일 상승", "change_rate": change_rate, "price_type": "close"},
        "stock_trend": None, "news": news or [],
        "summary": {"news_count": 1 if news else 0, "news_chunk_count": len(news or [])},
        "data_sources": {"stock": "toss", "news": "hybrid_search"},
    }


# 동일 기사의 청크를 본문 순서로 묶고 출처·URL을 한 번만 표시
def test_response_groups_article_chunks_and_shows_source():
    base = {"title": "실적 기사", "published_date": "2026-11-01", "publisher": "경제신문",
            "url": "https://example.test/article", "news_id": "article"}
    context = _context([dict(base, content="뒤 본문", chunk_index=2), dict(base, content="첫 본문", chunk_index=1)])
    before = deepcopy(context)
    answer = generate_stock_advice(context)
    assert answer.count("- 실적 기사:") == 1
    assert "첫 본문\n뒤 본문" in answer
    assert answer.count("https://example.test/article") == 1
    assert "경제신문 | 2026-11-01" in answer
    assert "관련 기사 1건" in answer
    assert "샘플" not in answer and "등락률" not in answer
    assert context == before


# 템플릿 출력은 발췌하되 원본 컨텍스트의 본문은 잘라내지 않음
def test_response_excerpt_does_not_truncate_context():
    news = [{"title": "기사", "content": "가" * 500, "published_date": "2026-11-01"}]
    context = _context(news)
    answer = generate_stock_advice(context)
    assert "가" * 300 + "..." in answer
    assert len(context["news"][0]["content"]) == 500


# 빈 실제 뉴스와 샘플 뉴스 표시를 구분하고 기존 summary 입력도 수용
def test_empty_news_and_sample_summary():
    assert "확인된 관련 뉴스는 없습니다." in generate_stock_advice(_context())
    context = _context([{"title": "샘플 기사", "summary": "샘플 요약", "published_date": "2026-11-01"}])
    context["data_sources"]["news"] = "sample"
    answer = generate_stock_advice(context)
    assert "샘플 뉴스 흐름은" in answer and "샘플 요약" in answer


# 상승·하락 질문에서도 내부 등락률 값은 노출하지 않고 인과관계를 단정하지 않음
@pytest.mark.parametrize("intent, rate, expected", [
    ("rise_reason", 1, "상승은 확인되지만"), ("rise_reason", -1, "뚜렷한 상승이 확인되지"),
    ("decline_reason", -1, "하락은 확인되지만"), ("decline_reason", 1, "뚜렷한 하락이 확인되지"),
])
def test_direction_intent_responses(intent, rate, expected):
    answer = generate_stock_advice(_context(intent=intent, change_rate=rate))
    assert expected in answer
    assert "등락률" not in answer and "%" not in answer


# 비교 기간이 다르면 강세 종목을 확정하지 않음
def test_comparison_requires_matching_periods():
    first, second = _context(), _context()
    first["stock_trend"] = {"stock_name": "삼성전자", "base_date": "2026-11-01", "recent_days": 5,
                            "recent_change_rate": 1}
    second["stock_trend"] = {"stock_name": "SK하이닉스", "base_date": "2026-11-02", "recent_days": 5,
                             "recent_change_rate": 2}
    assert "상대 흐름을 확정하지 않았습니다" in generate_stock_comparison_advice([first, second])

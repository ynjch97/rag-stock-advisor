from typing import Any

from src.data_loaders.sample_stock_loader import load_sample_stocks


ParsedQuestion = dict[str, Any]

BUY_OPINION_KEYWORDS = ["사도", "매수", "살까", "사야", "진입"]
DECLINE_REASON_KEYWORDS = ["하락", "떨어", "빠지", "급락", "약세"]


# 사용자 질문에서 종목명과 의도를 추출
def parse_stock_question(question: str) -> ParsedQuestion:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question must not be empty.")

    stock_name = _extract_stock_name(normalized_question)
    intent = _classify_intent(normalized_question)

    return {
        "original_question": question,  # 삼성전자 사도 될까?
        "stock_name": stock_name,       # 삼성전자
        "intent": intent,               # buy_opinion
    }


# sample_stocks.json에 있는 종목명 기준으로 질문에서 종목명 찾기
def _extract_stock_name(question: str) -> str:
    stock_names = _load_sample_stock_names()

    for stock_name in stock_names:
        if stock_name in question:
            return stock_name

    raise ValueError(f"Stock name not found in question: {question}")


# 질문 유형 기준으로 의도 분류
def _classify_intent(question: str) -> str:
    if any(keyword in question for keyword in BUY_OPINION_KEYWORDS):
        return "buy_opinion"

    if any(keyword in question for keyword in DECLINE_REASON_KEYWORDS):
        return "decline_reason"

    return "general_analysis"


# 긴 종목명을 먼저 검사해서 부분 매칭 충돌 가능성 줄임 (reverse=True, 내림차순)
def _load_sample_stock_names() -> list[str]:
    stocks = load_sample_stocks()
    stock_names = {stock["stock_name"] for stock in stocks}

    return sorted(stock_names, key=len, reverse=True)

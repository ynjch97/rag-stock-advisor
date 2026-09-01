from dataclasses import dataclass, asdict
from typing import Any

from src.data_loaders.sample_stock_loader import load_sample_stocks

BUY_OPINION_KEYWORDS = ["사도", "매수", "살까", "사야", "진입"]
DECLINE_REASON_KEYWORDS = ["하락", "떨어", "빠지", "급락", "약세"]

@dataclass(frozen=True)
class ParsedQuestion:
    original_question: str
    stock_name: str
    stock_code: str
    intent: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)
    
# 사용자 질문에서 종목명, 종목코드, 의도를 추출
def parse_stock_question(question: str) -> ParsedQuestion:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question must not be empty.")

    stock_info = _extract_stock_info(normalized_question) # 종목명/종목코드 가져오기
    intent = _classify_intent(normalized_question)        # 의도 추출하기

    return ParsedQuestion(
        original_question=question,  # 삼성전자 사도 될까?
        stock_name=stock_info["stock_name"], # 삼성전자
        stock_code=stock_info["stock_code"], # 005930
        intent=intent,               # buy_opinion
    )


# sample_stocks.json에 있는 종목명/종목코드 기준으로 질문에서 종목 찾기
def _extract_stock_info(question: str) -> dict[str, str]:
    stock_infos = _load_sample_stock_infos()

    for stock_info in stock_infos:
        if (
            stock_info["stock_name"] in question
            or stock_info["stock_code"] in question
        ):
            return stock_info

    raise ValueError(f"Stock name or stock code not found in question: {question}")


# 질문 유형 기준으로 의도 분류 (추후 LLM 으로 변경)
def _classify_intent(question: str) -> str:
    if any(keyword in question for keyword in BUY_OPINION_KEYWORDS):
        return "buy_opinion"

    if any(keyword in question for keyword in DECLINE_REASON_KEYWORDS):
        return "decline_reason"

    return "general_analysis"


# 긴 종목명을 먼저 검사해서 부분 매칭 충돌 가능성 줄임 (reverse=True, 내림차순)
def _load_sample_stock_infos() -> list[dict[str, str]]:
    stocks = load_sample_stocks()
    stock_infos = {
        (stock["stock_name"], stock["stock_code"])
        for stock in stocks
    }

    return [
        {"stock_name": stock_name, "stock_code": stock_code}
        for stock_name, stock_code in sorted(
            stock_infos,
            key=lambda stock_info: len(stock_info[0]), # 이름 없는 임시 함수 사용, 종목명 길이로 정렬
            reverse=True,
        )
    ]

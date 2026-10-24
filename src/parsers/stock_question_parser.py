from dataclasses import dataclass, asdict
import re

from src.collectors.stock_reference_collector import refresh_stock_references
from src.data_loaders.stock_reference_loader import load_stock_references
from src.domain.stock_models import DEFAULT_RECENT_DAYS

BUY_OPINION_KEYWORDS = ["사도", "매수", "살까", "사야", "진입"]
DECLINE_REASON_KEYWORDS = ["하락", "떨어", "빠지", "급락", "약세"]

@dataclass(frozen=True)
class ParsedQuestion:
    original_question: str
    stock_name: str
    stock_code: str
    intent: str
    recent_days: int = DEFAULT_RECENT_DAYS

    # 질문 분석 결과를 딕셔너리로 변환
    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)
    
# 사용자 질문에서 종목명, 종목코드, 의도를 추출
def parse_stock_question(
    question: str, *, stock_infos: list[dict[str, str]] | None = None,
) -> ParsedQuestion:
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question must not be empty.")

    recent_days = _extract_recent_days(normalized_question)
    stock_info = (
        _extract_stock_info(normalized_question) if stock_infos is None
        else _find_stock_info(normalized_question, stock_infos)
    )
    if stock_info is None:
        raise ValueError(f"Stock name or stock code not found in question: {question}")
    intent = _classify_intent(normalized_question)        # 의도 추출하기

    return ParsedQuestion(
        original_question=question,  # 삼성전자 사도 될까?
        stock_name=stock_info["stock_name"], # 삼성전자
        stock_code=stock_info["stock_code"], # 005930
        intent=intent,               # buy_opinion
        recent_days=recent_days,
    )


# 질문의 '최근 n거래일/최근 n일'을 거래일 수로 해석하고 미지정 시 5일 사용
def _extract_recent_days(question: str) -> int:
    match = re.search(r"최근\s*(\d+)\s*(?:거래일|일)", question)
    if match is None:
        return DEFAULT_RECENT_DAYS
    recent_days = int(match.group(1))
    if not 1 <= recent_days <= 199:
        raise ValueError("recent_days must be between 1 and 199 trading days.")
    return recent_days


# 기준정보에서 종목을 찾고 파일 누락 또는 매칭 실패 시 한 번만 갱신
def _extract_stock_info(question: str) -> dict[str, str]:
    refreshed = False
    try:
        stock_infos = load_stock_references()
    except FileNotFoundError:
        stock_infos = refresh_stock_references()
        refreshed = True

    stock_info = _find_stock_info(question, stock_infos)
    if stock_info is None and not refreshed:
        stock_info = _find_stock_info(question, refresh_stock_references())
    if stock_info is not None:
        return stock_info

    raise ValueError(f"Stock name or stock code not found in question: {question}")


# 질문 유형 기준으로 의도 분류 (추후 LLM 으로 변경)
def _classify_intent(question: str) -> str:
    if any(keyword in question for keyword in BUY_OPINION_KEYWORDS):
        return "buy_opinion"

    if any(keyword in question for keyword in DECLINE_REASON_KEYWORDS):
        return "decline_reason"

    return "general_analysis"


# 종목코드를 우선 확인하고 긴 종목명부터 검사하여 부분 매칭 충돌 방지
def _find_stock_info(
    question: str, stock_infos: list[dict[str, str]],
) -> dict[str, str] | None:
    codes = set(re.findall(r"(?<![A-Za-z0-9])[A-Za-z0-9]{6}(?![A-Za-z0-9])", question))
    for stock_info in stock_infos:
        if stock_info["stock_code"] in codes:
            return stock_info
    for stock_info in sorted(
        stock_infos, key=lambda stock: len(stock["stock_name"]), reverse=True,
    ):
        if stock_info["stock_name"] in question:
            return stock_info
    return None

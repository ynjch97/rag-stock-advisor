from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.domain.stock_models import DEFAULT_RECENT_DAYS


@dataclass(frozen=True)
class ParsedStock:
    mention: str
    stock_name: str
    stock_code: str
    role: str = "subject"


@dataclass(frozen=True)
class QueryPeriod:
    kind: Literal["default", "trading_days", "date_range"] = "default"
    recent_days: int | None = DEFAULT_RECENT_DAYS
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class ParsedQuestion:
    original_question: str
    stocks: tuple[ParsedStock, ...]
    intent: str
    period: QueryPeriod = field(default_factory=QueryPeriod)
    needs_clarification: bool = False
    clarification_question: str | None = None
    openai_call_attempts: int = 0

    # 기존 단일 종목 호출부에서 첫 종목명 참조 지원
    @property
    def stock_name(self) -> str:
        return self.stocks[0].stock_name

    # 검증된 첫 종목코드 참조 지원
    @property
    def stock_code(self) -> str:
        return self.stocks[0].stock_code

    # 날짜 범위의 임의 거래일 환산 방지
    @property
    def recent_days(self) -> int:
        if self.period.recent_days is None:
            raise ValueError("A date range cannot be treated as a trading-day count.")
        return self.period.recent_days

    # 질문·종목·기간을 컨텍스트 전달용 딕셔너리로 변환
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

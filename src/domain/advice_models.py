from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.domain.stock_models import DEFAULT_RECENT_DAYS


DEFAULT_DISCOVERY_RESULTS = 3
MAX_DISCOVERY_RESULTS = 10


# 대상 종목 정보
@dataclass(frozen=True)
class ParsedStock:
    mention: str
    stock_name: str
    stock_code: str
    role: str = "subject"


# 기간 정보
@dataclass(frozen=True)
class QueryPeriod:
    kind: Literal["default", "trading_days", "date_range"] = "default"
    recent_days: int | None = DEFAULT_RECENT_DAYS
    start_date: str | None = None
    end_date: str | None = None


# 종목 추천 조건 (요청 개수는 임의로 상한에 맞춰 잘라내지 않음)
@dataclass(frozen=True)
class StockDiscoveryCriteria:
    theme: str
    requested_count: int = DEFAULT_DISCOVERY_RESULTS


# 종목 추천 근거와 출처 (기사·공시 등에서 확인)
@dataclass(frozen=True)
class StockDiscoveryEvidence:
    title: str
    url: str
    content: str
    publisher: str | None = None
    published_date: str | None = None
    excerpt: str | None = None  # 사업 연관성을 확인한 실제 인용 구간 (content는 전체 본문 보존)


# 종목 검색 후보 (코드 매핑과 사업 검증은 후속 발굴 서비스가 수행)
@dataclass(frozen=True)
class StockDiscoveryCandidate:
    stock_name: str
    selection_reason: str
    evidence: tuple[StockDiscoveryEvidence, ...] = ()
    stock_code: str | None = None
    business_verified: bool = False

    # 원문 근거를 포함한 후보 정보를 딕셔너리로 변환
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 질문 분석
@dataclass(frozen=True)
class ParsedQuestion:
    original_question: str
    stocks: tuple[ParsedStock, ...]
    intent: str
    period: QueryPeriod = field(default_factory=QueryPeriod)
    needs_clarification: bool = False
    clarification_question: str | None = None
    openai_call_attempts: int = 0
    discovery: StockDiscoveryCriteria | None = None
    discovered_candidates: tuple[StockDiscoveryCandidate, ...] = ()

    # 질문에 명시된 종목과 발굴 후보를 구분한 채 실제 분석 대상을 반환
    @property
    def analysis_stocks(self) -> tuple[ParsedStock, ...]:
        if self.intent != "stock_discovery":
            return self.stocks
        return tuple(
            ParsedStock(candidate.stock_name, candidate.stock_name, candidate.stock_code, "discovered")
            for candidate in self.discovered_candidates
            if candidate.business_verified and candidate.stock_code
        )

    # 기존 단일 종목 호출부에서 첫 종목명 참조 지원
    @property
    def stock_name(self) -> str:
        if not self.stocks:
            raise ValueError("No specified stock; discovery questions require candidate discovery first.")
        return self.stocks[0].stock_name

    # 검증된 첫 종목코드 참조 지원
    @property
    def stock_code(self) -> str:
        if not self.stocks:
            raise ValueError("No specified stock; discovery questions require candidate discovery first.")
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


# Controller가 선택한 다음 작업과 대상, 이유 (실행 및 횟수 제한은 workflow·Agent 담당)
@dataclass(frozen=True)
class AgentDecision:
    # 응답 생성 / 주가 보충 / 뉴스 수집·인덱싱·재검색 / 기존 인덱스 재검색 / 종료
    action: Literal["respond", "collect_stock", "collect_news", "search_news", "discover_stocks", "stop"]
    reason: str
    stock: ParsedStock | None = None  # 종목별 작업의 대상 (발굴·전체 응답·종료이면 생략)
    search_query: str | None = None  # 검색 표현을 바꾸는 경우 사용할 질문
    news_start: Literal[1, 31] | None = None  # 첫 페이지 또는 추가 페이지

    # 중첩된 대상 종목을 포함하여 실행 계획을 딕셔너리로 변환
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 추가 작업 시도는 실패해도 기록하여 동일 작업 반복과 예산 초과를 방지
@dataclass(frozen=True)
class AgentExecutionRecord:
    action: str
    stock_code: str | None
    success: bool
    reason: str
    search_query: str | None = None
    news_start: int | None = None
    new_articles: int = 0
    next_start: int | None = None  # 다음 페이지 시작 위치

    # LLM에 전달할 실행 이력을 직렬화
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

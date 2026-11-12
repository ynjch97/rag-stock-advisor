from datetime import date, datetime, timedelta, timezone

from openai import OpenAI

from src.collectors.stock_reference_collector import refresh_stock_references
from src.config.settings import Settings, settings
from src.data_loaders.stock_reference_loader import load_stock_references
from src.domain.advice_models import (
    DEFAULT_DISCOVERY_RESULTS, MAX_DISCOVERY_RESULTS, ParsedQuestion, ParsedStock, QueryPeriod, StockDiscoveryCriteria,
)
from src.domain.stock_models import DEFAULT_RECENT_DAYS
from src.parsers.llm_question_analyzer import AnalyzedDiscovery, AnalyzedPeriod, AnalyzedStock, analyze_stock_question

# LLM으로 질문을 해석하고 기준정보 검증 후 ParsedQuestion 반환
def parse_stock_question(
    question: str, *, stock_infos: list[dict[str, str]] | None = None,
    app_settings: Settings = settings, client: OpenAI | None = None,
    reference_date: date | None = None,
) -> ParsedQuestion:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    today = reference_date or datetime.now(timezone(timedelta(hours=9))).date()
    analysis = analyze_stock_question(question.strip(), today, app_settings, client)
    period = _validate_period(analysis.period, today)
    discovery, discovery_question = _parse_discovery_criteria(analysis.intent, analysis.discovery)
    if (analysis.needs_clarification or discovery_question
            or not analysis.stocks and analysis.intent != "stock_discovery"):
        return ParsedQuestion(question, (), analysis.intent, period, True,
                              analysis.clarification_question or discovery_question or "분석할 종목과 기간을 구체적으로 알려주세요.",
                              openai_call_attempts=analysis.openai_call_attempts, discovery=discovery)
    if not analysis.stocks:
        return ParsedQuestion(question, (), analysis.intent, period,
                              openai_call_attempts=analysis.openai_call_attempts, discovery=discovery)
    # 발굴 질문도 임의 후보를 질문 속 종목처럼 주입할 수 없으며 검증 전 외부 조회도 차단
    # "질문에 나온 종목"과 "앞으로 검색해서 찾을 후보"를 구분
    for item in analysis.stocks:
        if not item.mention.strip() or _normalize_name(item.mention) not in _normalize_name(question):
            raise ValueError("LLM의 종목 원문 표현이 사용자 질문에 존재하지 않습니다.")
    if stock_infos is None:
        try:
            stock_infos = load_stock_references()
        except FileNotFoundError:
            stock_infos = refresh_stock_references()
        else:
            if any(_resolve_stock(item, stock_infos) is None for item in analysis.stocks):
                stock_infos = refresh_stock_references()
    resolved = []
    for item in analysis.stocks:
        match = _resolve_stock(item, stock_infos)
        if match is None:
            return ParsedQuestion(question, tuple(resolved), analysis.intent, period, True,
                                  f"'{item.mention}'의 정식 종목명이나 종목코드를 알려주세요.",
                                  openai_call_attempts=analysis.openai_call_attempts, discovery=discovery)
        if match.stock_code not in {stock.stock_code for stock in resolved}:
            resolved.append(match)
    if analysis.intent == "comparison_reason" and len(resolved) < 2:
        return ParsedQuestion(question, tuple(resolved), analysis.intent, period, True, "비교할 두 종목을 알려주세요.",
                              openai_call_attempts=analysis.openai_call_attempts)
    return ParsedQuestion(question, tuple(resolved), analysis.intent, period,
                          openai_call_attempts=analysis.openai_call_attempts, discovery=discovery)


# 발굴 조건만 구조화하며 후보를 생성하지 않음; 개수 초과는 축소 대신 확인 요청
def _parse_discovery_criteria(
    intent: str, discovery: AnalyzedDiscovery | None,
) -> tuple[StockDiscoveryCriteria | None, str | None]:
    if intent != "stock_discovery":
        if discovery is not None:
            raise ValueError("Only stock_discovery questions may contain discovery criteria.")
        return None, None
    if discovery is None or not discovery.theme or not discovery.theme.strip():
        return None, "발굴할 종목의 업종·테마 또는 조건을 알려주세요."
    count = discovery.requested_count if discovery.requested_count is not None else DEFAULT_DISCOVERY_RESULTS
    criteria = StockDiscoveryCriteria(discovery.theme.strip(), count)
    if count > MAX_DISCOVERY_RESULTS:
        return criteria, f"현재 종목 발굴은 최대 {MAX_DISCOVERY_RESULTS}개까지 가능합니다. 요청 개수를 확인해주세요."
    return criteria, None


# 표기 차이만 정규화하며 수동 별칭 목록은 사용하지 않음
def _normalize_name(value: str) -> str:
    return "".join(value.split()).casefold()


# 코드 원문 또는 정식 이름 후보가 기준정보에서 유일하게 일치할 때 코드 확정
def _resolve_stock(item: AnalyzedStock, records: list[dict[str, str]]) -> ParsedStock | None:
    matches = [record for record in records if record["stock_code"] == item.mention.strip()]
    if not matches:
        matches = [record for record in records if _normalize_name(record["stock_name"]) ==
                   _normalize_name(item.stock_name or item.mention)]
    unique = {record["stock_code"]: record for record in matches}
    if len(unique) != 1:
        return None
    record = next(iter(unique.values()))
    return ParsedStock(item.mention, record["stock_name"], record["stock_code"], item.role)


# 기간 기본값·날짜 순서를 검증하고 달력 기간을 거래일로 임의 환산하지 않음
def _validate_period(period: AnalyzedPeriod, today: date) -> QueryPeriod:
    if period.kind in ("default", "trading_days"):
        if period.start_date is not None or period.end_date is not None:
            raise ValueError("거래일 기간과 날짜 범위를 함께 지정할 수 없습니다.")
        if period.kind == "default" and period.recent_days is not None:
            raise ValueError("미지정 기간은 Python 기본값을 사용해야 합니다.")
        if period.kind == "trading_days" and period.recent_days is None:
            raise ValueError("거래일 수가 누락되었습니다.")
        return QueryPeriod(period.kind, period.recent_days or DEFAULT_RECENT_DAYS)
    if period.recent_days is not None:
        raise ValueError("날짜 범위에는 거래일 수를 함께 지정할 수 없습니다.")
    try:
        start, end = date.fromisoformat(period.start_date), date.fromisoformat(period.end_date)
    except (TypeError, ValueError):
        raise ValueError("유효한 시작일과 종료일이 필요합니다.") from None
    if start > end or end > today:
        raise ValueError("조회 기간의 날짜 순서 또는 미래 날짜를 확인해야 합니다.")
    return QueryPeriod("date_range", None, start.isoformat(), end.isoformat())

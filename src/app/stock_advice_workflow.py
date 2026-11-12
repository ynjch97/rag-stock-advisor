from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time

import httpx

from src.collectors.stock_collector import collect_stock_price
from src.config.settings import Settings, settings
from src.config.paths import BM25_TOKENS_PATH, PROJECT_ROOT, RAW_NEWS_DIR, get_raw_price_path
from src.collectors.news_collector import summarize_news_page
from src.data_loaders.sample_news_loader import load_recent_sample_news
from src.data_loaders.sample_stock_loader import load_latest_sample_stock, load_sample_stocks
from src.parsers.stock_question_parser import parse_stock_question
from src.domain.stock_models import DailyStock
from src.domain.advice_models import AgentExecutionRecord, ParsedQuestion, ParsedStock
from src.agents.stock_advisor_agent import decide_next_action
from src.app.rag_indexing_workflow import collect_and_index_news
from src.preprocessing.stock_preprocessor import load_latest_stock_trend
from src.search.hybrid_search import search_news_hybrid
from src.services.stock_advice_service import generate_stock_advice, generate_stock_comparison_advice
from src.services.stock_context_service import StockContext, attach_agent_results, build_stock_context
from src.services.execution_budget import deadline_http_client, remaining_timeout


TOOL_ERRORS = (ValueError, OSError, TimeoutError, httpx.HTTPError)
KST = timezone(timedelta(hours=9))


# provider에 따라 샘플 또는 실제 주가·추세를 확보한 뒤 분석 응답 생성
def run_stock_advice_workflow(question: str, app_settings: Settings = settings) -> str:
    provider = (app_settings.stock_api_provider or "").lower()
    if provider not in ("sample", "toss"):
        raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")
    # sample/toss는 주가 제공자만 구분하며 질문 분석은 항상 OpenAI 사용
    stock_infos = None
    if provider == "sample":
        stock_infos = [
            {"stock_name": stock["stock_name"], "stock_code": stock["stock_code"]}
            for stock in load_sample_stocks()
        ]
    parsed_question = parse_stock_question(question, stock_infos=stock_infos, app_settings=app_settings)
    if parsed_question.needs_clarification:
        return parsed_question.clarification_question or "분석할 종목과 기간을 확인해주세요."
    if parsed_question.period.kind == "date_range":
        return (
            f"질문의 기간은 {parsed_question.period.start_date}부터 {parsed_question.period.end_date}까지입니다. "
            "현재 조회는 거래일 수 기준입니다. 최근 몇 거래일을 분석할지 알려주세요."
        )
    deadline = time.monotonic() + app_settings.agent_timeout_seconds
    contexts = []
    for stock in parsed_question.stocks:
        if provider == "sample":
            contexts.append(_build_analysis_context(parsed_question, stock, app_settings))
            continue
        try:
            remaining_timeout(deadline)
            contexts.append(_build_analysis_context(parsed_question, stock, app_settings, deadline=deadline))
        except TOOL_ERRORS as error:
            context = build_stock_context(parsed_question, {
                "stock_name": stock.stock_name, "stock_code": stock.stock_code, "current_price": None,
                "base_date": None, "trend": "확인 불가", "change_rate": None, "source": "toss",
            }, [], news_source="hybrid_search", issues=[_tool_error("주가 조회", error)])
            context.update(news_period=None, news_collections=[], availability={
                "stock": False, "stock_retryable": _retryable(error), "news_retryable": True,
            })
            contexts.append(context)
    if provider == "toss":
        _run_agent_loop(parsed_question, contexts, app_settings, deadline)
    if len(contexts) == 1:
        return generate_stock_advice(contexts[0])
    return generate_stock_comparison_advice(contexts)


# 검증된 종목 하나의 가격·추세·뉴스를 기존 컨텍스트 구조로 구성
def _build_analysis_context(
    parsed_question: ParsedQuestion, stock: ParsedStock, app_settings: Settings,
    *, deadline: float | None = None, previous_context: StockContext | None = None,
) -> StockContext:
    provider = (app_settings.stock_api_provider or "").lower()
    trend_data = None
    news_source = "sample"
    issues, collections, period = [], [], None
    news_retryable = True
    if provider == "sample":
        stock_data = load_latest_sample_stock(stock.stock_name)
        news_data = load_recent_sample_news(stock.stock_name)
    else:
        # 수집 함수 내부에서 완료 일봉 저장과 필요한 추세 전처리까지 수행
        with deadline_http_client(deadline, app_settings.agent_request_timeout_seconds) as connection:
            price = collect_stock_price(
                stock.stock_name, stock.stock_code, http_client=connection,
                app_settings=app_settings, recent_days=parsed_question.recent_days,
            )
        trend_data = load_latest_stock_trend(
            price.stock_code, parsed_question.recent_days, base_date=price.base_date,
        )
        if isinstance(price, DailyStock) and trend_data.base_date != price.base_date:
            raise ValueError("Daily price and stock trend dates must match.")
        if not isinstance(price, DailyStock) and trend_data.base_date >= price.base_date:
            raise ValueError("Realtime analysis requires a previously completed day's trend.")
        stock_data = asdict(price)
        stock_data.update(
            current_price=price.close if isinstance(price, DailyStock) else price.current_price,
            price_type="close" if isinstance(price, DailyStock) else "realtime",
            source="toss",
        )
        start_date, end_date = _get_news_search_period(
            stock.stock_code, parsed_question.recent_days, trend_data.base_date, price.base_date,
        )
        period = {"start_date": start_date, "end_date": end_date}
        news_data = []
        try:
            collections = _read_news_collections(stock.stock_code, app_settings)
            if previous_context is None:
                if _missing_news_index(app_settings):
                    issues.append("기존 뉴스 인덱스에 검색할 데이터가 없습니다.")
                else:
                    news_data = search_news_hybrid(
                        parsed_question.original_question, stock_code=stock.stock_code,
                        **period, app_settings=app_settings, deadline=deadline,
                    )
            else:
                # 주가 보충 결정은 주가만 재조회하며 기존 뉴스 중 새 기간에 맞는 근거 유지
                news_data = [item for item in previous_context["news"]
                             if start_date <= item["published_date"] <= end_date]
        except TOOL_ERRORS as error:
            news_retryable = isinstance(error, FileNotFoundError) and _missing_news_index(app_settings)
            news_retryable = news_retryable or _retryable(error) and not isinstance(error, FileNotFoundError)
            issues.append(_tool_error("뉴스 검색", error))
        news_source = "hybrid_search"

    context = build_stock_context(
        parsed_question=parsed_question,
        stock_data=stock_data,
        news_data=news_data,
        trend_data=trend_data,
        news_source=news_source,
        issues=issues,
    )
    context.update(news_period=period, news_collections=collections, checked_at=datetime.now(KST).isoformat(),
                   availability={"stock": True, "stock_retryable": True, "news_retryable": news_retryable})
    return context


# 비밀키·응답 본문을 노출하지 않고 작업 단계와 오류 종류만 사용자에게 전달
def _tool_error(stage: str, error: Exception) -> str:
    status = f", HTTP {error.response.status_code}" if isinstance(error, httpx.HTTPStatusError) else ""
    return f"{stage} 실패 ({type(error).__name__}{status})."


# 인증·입력·손상 오류는 자동 재시도하지 않고 연결 장애·429·서버 장애만 보충 허용
def _retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, (httpx.RequestError, FileNotFoundError, TimeoutError))


# 인덱스 최초 부재와 정상적으로 0건 수집된 상태만 자동 수집 대상으로 인정
def _missing_news_index(app_settings: Settings) -> bool:
    directory = PROJECT_ROOT / app_settings.vector_db_path
    path = directory / "stock_advisor_metadata.json"
    if not path.exists():
        return not (directory / "stock_advisor.index").exists() and not BM25_TOKENS_PATH.exists()
    try:
        return (json.loads(path.read_text(encoding="utf-8")).get("chunk_count") == 0
                and not (directory / "stock_advisor.index").exists())
    except (OSError, ValueError, AttributeError):
        return False


# 성공한 인덱싱에 반영된 수집 이력만 해당 종목의 최신성 판단에 사용
def _read_news_collections(stock_code: str, app_settings: Settings) -> list[dict]:
    path = PROJECT_ROOT / app_settings.vector_db_path / "stock_advisor_metadata.json"
    if not path.exists():
        return []
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("collections", []), list):
        raise ValueError("Invalid saved news collection history.")
    collections = metadata.get("collections", [])
    for item in collections:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in (
            "stock_code", "start_date", "end_date", "collected_at",
        )):
            raise ValueError("Invalid saved news collection fields.")
        collected_at = datetime.fromisoformat(item["collected_at"])
        if (collected_at.utcoffset() is None or date.fromisoformat(item["start_date"]) > date.fromisoformat(item["end_date"])
                or item.get("start", 1) not in (1, 31)):
            raise ValueError("Invalid saved news collection dates or page.")
    result = []
    for item in collections:
        if item["stock_code"] != stock_code:
            continue
        # 이전 이력은 대응 raw에서 페이지 상태만 복원; API 호출·재임베딩은 하지 않음
        item = dict(item)
        chunk_file = item.get("chunk_file")
        if "next_start" not in item and isinstance(chunk_file, str) and chunk_file.endswith("_chunks.json"):
            raw_path = RAW_NEWS_DIR / (Path(chunk_file).name[:-len("_chunks.json")] + ".json")
            if raw_path.exists():
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                item.update(summarize_news_page(raw, item["start_date"], item["end_date"], item.get("start", 1)))
        result.append(item)
    return result


# 서버가 현재 예산·성공/실패 이력으로 실행 가능한 도구만 허용 (LLM이 제한을 늘릴 수 없음)
# 남은 횟수·실행 이력·데이터 상태를 확인 -> 지금 실행해도 되는 작업 목록을 만드는 함수
def _allowed_agent_actions(parsed: ParsedQuestion, contexts: list[StockContext], history: list[AgentExecutionRecord],
                           indexing_calls: int, app_settings: Settings) -> list[dict]:
    allowed = [{"action": "stop"}]
    # 최소 근거조차 없으면 LLM이 충분하다고 선언하지 못하도록 하며 부분 응답은 stop으로 처리
    if all(context["availability"]["stock"] and context["news"] for context in contexts):
        allowed.insert(0, {"action": "respond"})
    now = datetime.now(KST)
    for stock, context in zip(parsed.stocks, contexts):
        code, availability = stock.stock_code, context["availability"]
        records = [record for record in history if record.stock_code == code]
        stale_price = context["stock"].get("price_type") == "realtime" and (
            now - datetime.fromisoformat(context["checked_at"])).total_seconds() >= 60
        if ((not availability["stock"] or stale_price) and availability["stock_retryable"]
                and sum(record.action == "collect_stock" for record in records) < app_settings.agent_max_stock_refreshes):
            allowed.append({"action": "collect_stock", "stock_code": code})
        period = context.get("news_period")
        if not period or not availability["news_retryable"]:
            continue
        allowed.append({"action": "search_news", "stock_code": code})
        if indexing_calls >= app_settings.agent_max_indexing_calls:
            continue
        pages = {record.news_start for record in records if record.action == "collect_news"}
        recent = [item for item in context["news_collections"]
                  if item["start_date"] <= period["start_date"] and item["end_date"] >= period["end_date"]
                  and 0 <= (now - datetime.fromisoformat(item["collected_at"])).total_seconds()
                  < app_settings.agent_news_refresh_seconds]
        pages.update(item.get("start", 1) for item in recent)
        if len(pages) >= app_settings.agent_max_news_pages:
            continue
        if 1 not in pages:
            allowed.append({"action": "collect_news", "stock_code": code, "news_start": 1})
        elif 31 not in pages and (any(item.get("next_start") == 31 for item in recent)
                                 or (not any("next_start" in item for item in recent)
                                     and not any(item.get("article_count") == 0 for item in recent))):
            allowed.append({"action": "collect_news", "stock_code": code, "news_start": 31})
    return allowed


# 점수·재생성된 ID가 아닌 가격과 실제 뉴스 내용으로 추가 근거 유무 판단
def _evidence_signature(context: StockContext) -> tuple:
    stock = context["stock"]
    return (stock.get("base_date"), stock.get("current_price"), stock.get("trend"),
            tuple(sorted((item.get("url", ""), item["content"]) for item in context["news"])))


# LLM 판단 -> 허용 작업만 실행 -> 결과 재평가
# 결정, 페이지·인덱싱, 시간 제한을 코드로 강제
def _run_agent_loop(parsed: ParsedQuestion, contexts: list[StockContext], app_settings: Settings, deadline: float) -> None:
    history, notes, indexing_calls = [], [], 0
    for _ in range(app_settings.agent_max_decisions):
        try:
            remaining_timeout(deadline)
            allowed = _allowed_agent_actions(parsed, contexts, history, indexing_calls, app_settings)
            decision = decide_next_action(parsed, contexts, history, allowed, app_settings=app_settings, deadline=deadline)
            code = decision.stock.stock_code if decision.stock is not None else None
            if not any(item["action"] == decision.action and item.get("stock_code") == code
                       and (decision.action != "collect_news" or item["news_start"] == decision.news_start)
                       for item in allowed):
                raise ValueError("Agent action exceeds the allowed budget.")
            if decision.action in ("respond", "stop"):
                history.append(AgentExecutionRecord(decision.action, None, True, decision.reason))
                if decision.action == "stop":
                    notes.append(f"추가 조사 중단: {decision.reason}")
                break
            remaining_timeout(deadline)
            position = next(i for i, stock in enumerate(parsed.stocks) if stock.stock_code == code)
            context, stock = contexts[position], parsed.stocks[position]
            query = (decision.search_query or parsed.original_question).strip()
            if decision.action == "search_news" and (query.casefold() == parsed.original_question.strip().casefold()
                or any(record.stock_code == code and record.search_query and record.search_query.casefold() == query.casefold()
                       for record in history)):
                notes.append("동일한 검색 질문의 반복을 차단했습니다.")
                break
            before, new_articles = _evidence_signature(context), 0
            try:
                if decision.action == "collect_stock":
                    contexts[position] = _build_analysis_context(parsed, stock, app_settings,
                                                                deadline=deadline, previous_context=context)
                else:
                    if decision.action == "collect_news":
                        indexing_calls += 1  # 실패한 인덱싱 시도도 예산에 포함
                        result = collect_and_index_news(stock, **context["news_period"], start=decision.news_start,
                                                        app_settings=app_settings, deadline=deadline)
                        new_articles = result["new_articles"]
                    remaining_timeout(deadline)
                    news = ([] if _missing_news_index(app_settings) else
                            search_news_hybrid(query, stock_code=code, **context["news_period"],
                                               app_settings=app_settings, deadline=deadline))
                    context["news"] = news
                    context["summary"] = {"news_count": len({item["url"] for item in news}), "news_chunk_count": len(news)}
                    context["news_collections"] = _read_news_collections(code, app_settings)
                    context["issues"] = [issue for issue in context["issues"]
                                         if not issue.startswith(("뉴스 검색 실패", "기존 뉴스 인덱스"))]
                history.append(AgentExecutionRecord(decision.action, code, True, decision.reason, query,
                                                    decision.news_start, new_articles))
            except TOOL_ERRORS as error:
                reason = _tool_error("추가 " + decision.action, error)
                history.append(AgentExecutionRecord(decision.action, code, False, reason, query, decision.news_start))
                contexts[position]["issues"].append(reason)
                contexts[position]["availability"]["stock_retryable" if decision.action == "collect_stock"
                                                     else "news_retryable"] = False
                continue
            if before == _evidence_signature(contexts[position]) and new_articles == 0:
                # 최신 기사만 받은 첫 페이지는 목표 기간에 도달하지 못한 것일 수 있음
                if (decision.action == "collect_news" and result.get("collection", {}).get("next_start") == 31
                    and any(
                    item["action"] == "collect_news" and item.get("stock_code") == code
                    for item in _allowed_agent_actions(parsed, contexts, history, indexing_calls, app_settings)
                )):
                    continue
                notes.append("추가 기사·근거가 늘지 않아 반복 조사를 종료했습니다.")
                break
        except TOOL_ERRORS as error:
            notes.append(_tool_error("Agent 처리", error))
            break
    else:
        notes.append(f"Agent 결정 횟수 상한({app_settings.agent_max_decisions}회)에 도달했습니다.")
    attach_agent_results(contexts, history, notes)


# 수집된 완료 일봉으로 최근 n거래일의 시작일 결정 (장중 뉴스는 당일까지 포함)
def _get_news_search_period(
    stock_code: str, recent_days: int, completed_date: str, price_date: str,
) -> tuple[str, str]:
    with get_raw_price_path(stock_code).open("r", encoding="utf-8") as file:
        records = json.load(file)
    cutoff, end = date.fromisoformat(completed_date), date.fromisoformat(price_date)
    days = sorted({
        date.fromisoformat(record["base_date"]) for record in records
        if record["stock_code"] == stock_code and date.fromisoformat(record["base_date"]) <= cutoff
    })
    if len(days) < recent_days or days[-1] != cutoff or end < cutoff:
        raise ValueError("Completed daily stocks are insufficient for the news search period.")
    return days[-recent_days].isoformat(), end.isoformat()

from dataclasses import replace
from datetime import datetime
import json
from unittest.mock import ANY, Mock

import httpx
import pytest

from src.app import stock_advice_workflow as workflow
from src.app.stock_advice_workflow import run_stock_advice_workflow
from src.collectors.stock_collector import collect_stock_price
from src.config.settings import settings
from src.domain.stock_models import DailyStock, RealtimeStock, StockTrend
from src.domain.advice_models import AgentDecision, ParsedQuestion, ParsedStock, QueryPeriod, StockDiscoveryCriteria
from src.parsers.llm_question_analyzer import QuestionAnalysis


# 기존 workflow 테스트는 OpenAI 응답만 모의하여 외부 호출 방지
@pytest.fixture(autouse=True)
def question_analysis(monkeypatch):
    monkeypatch.setattr(workflow, "decide_next_action", Mock(return_value=AgentDecision("respond", "테스트 근거 확보")))
    cases = {
        "삼성전자 사도 될까?": ("삼성전자", "buy_opinion", None),
        "삼성전자 전망": ("삼성전자", "general_analysis", None),
        "삼성전자 최근 3거래일 사도 될까?": ("삼성전자", "buy_opinion", 3),
        "두산에너빌리티 최근 3거래일 사도 될까?": ("두산에너빌리티", "buy_opinion", 3),
    }

    # 정해진 질문의 분석 결과를 반환
    def analyze(question, *args):
        name, intent, days = cases[question]
        return QuestionAnalysis.model_validate({
            "discovery": None,
            "stocks": [{"mention": name, "stock_name": name, "role": "subject"}],
            "intent": intent, "period": {"kind": "default" if days is None else "trading_days",
                "recent_days": days, "start_date": None, "end_date": None},
            "needs_clarification": False, "clarification_question": None,
        })

    monkeypatch.setattr("src.parsers.stock_question_parser.analyze_stock_question", analyze)


# 실제 뉴스 API·임베딩 호출을 막고 모의 수집 결과에 맞는 저장 일봉 제공
@pytest.fixture(autouse=True)
def saved_daily_prices_and_news(tmp_path, monkeypatch):
    days = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19",
            "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
    for code in ("005930", "000660", "034020"):
        (tmp_path / f"{code}_daily.json").write_text(json.dumps([
            {"stock_code": code, "base_date": day} for day in days
        ]), encoding="utf-8")
    monkeypatch.setattr(workflow, "get_raw_price_path", lambda code: tmp_path / f"{code}_daily.json")
    search = Mock(return_value=[])
    monkeypatch.setattr(workflow, "search_news_hybrid", search)
    return search


def test_run_stock_advice_workflow_generates_answer(monkeypatch, saved_daily_prices_and_news):
    monkeypatch.setattr(
        "src.parsers.stock_question_parser.load_stock_references",
        lambda: [{"stock_name": "삼성전자", "stock_code": "005930", "market": "KOSPI"}],
    )
    collector = Mock(side_effect=AssertionError("Sample workflow must not call the API."))
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    monkeypatch.setattr(
        "src.parsers.stock_question_parser.refresh_stock_references",
        Mock(side_effect=AssertionError("Sample workflow must not refresh references.")),
    )
    answer = run_stock_advice_workflow(
        "삼성전자 사도 될까?", replace(settings, stock_api_provider="sample"),
    )

    assert "[삼성전자 간단 분석]" in answer
    assert "현재가는" in answer
    assert "뉴스 흐름은" in answer
    assert "투자 참고용" in answer
    assert "등락률" not in answer
    assert "%" not in answer
    collector.assert_not_called()
    saved_daily_prices_and_news.assert_not_called()


# 실제 조회 분기의 현재가·종가·추세 전달과 등락률 비노출 검증
@pytest.mark.parametrize("realtime", [True, False])
def test_toss_workflow_passes_price_and_trend(monkeypatch, realtime, saved_daily_prices_and_news):
    monkeypatch.setattr(
        "src.parsers.stock_question_parser.load_stock_references",
        lambda: [{"stock_name": "두산에너빌리티", "stock_code": "034020", "market": "KOSPI"}],
    )
    price = (
        RealtimeStock("두산에너빌리티", "034020", 58900, -3.76, "2026-10-23")
        if realtime else DailyStock(
            "두산에너빌리티", "034020", "2026-10-23", 60000, 61000, 58000, 58900, 1000, -3.76, -2300,
        )
    )
    trend = StockTrend(
        "두산에너빌리티", "034020", "2026-10-22" if realtime else "2026-10-23",
        3, -5.5, 1, 2, "최근 3거래일 하락",
    )
    collector = Mock(return_value=price)
    loader = Mock(return_value=trend)
    generator = Mock(wraps=workflow.generate_stock_advice)
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    monkeypatch.setattr(workflow, "load_latest_stock_trend", loader)
    monkeypatch.setattr(workflow, "generate_stock_advice", generator)
    monkeypatch.setattr(workflow, "load_latest_sample_stock", Mock(side_effect=AssertionError("No sample price fallback")))
    config = replace(settings, stock_api_provider="toss")
    answer = run_stock_advice_workflow("두산에너빌리티 최근 3거래일 사도 될까?", config)
    collector.assert_called_once_with("두산에너빌리티", "034020", app_settings=config, recent_days=3, http_client=ANY)
    loader.assert_called_once_with("034020", 3, base_date="2026-10-23")
    context = generator.call_args.args[0]
    assert context["stock"]["current_price"] == 58900
    assert context["stock"]["change_rate"] == -3.76
    assert context["stock_trend"]["recent_change_rate"] == -5.5
    assert context["question"]["recent_days"] == 3
    assert context["news"] == []
    assert context["data_sources"] == {"stock": "toss", "news": "hybrid_search"}
    saved_daily_prices_and_news.assert_called_once_with(
        "두산에너빌리티 최근 3거래일 사도 될까?", stock_code="034020",
        start_date="2026-10-20" if realtime else "2026-10-21",
        end_date="2026-10-23", app_settings=config,
        deadline=ANY,
    )
    assert "최근 3거래일 하락" in answer
    assert "등락률" not in answer and "%" not in answer
    assert ("현재가는" if realtime else "2026-10-23 종가는") in answer


# 미지원 provider는 질문 분석이나 API 호출 전에 명시적으로 거부
def test_workflow_rejects_unsupported_provider(monkeypatch):
    parser = Mock()
    monkeypatch.setattr(workflow, "parse_stock_question", parser)
    with pytest.raises(ValueError, match="Unsupported STOCK_API_PROVIDER"):
        run_stock_advice_workflow("삼성전자", replace(settings, stock_api_provider="other"))
    parser.assert_not_called()


# 실제 시세 조회 실패를 샘플 값으로 대체하지 않고 호출자에게 전달
def test_toss_workflow_does_not_fallback_on_api_error(monkeypatch):
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=ParsedQuestion(
        "삼성전자 전망", (ParsedStock("삼성전자", "삼성전자", "005930"),), "general_analysis",
    )))
    monkeypatch.setattr(workflow, "collect_stock_price", Mock(side_effect=httpx.ConnectError("offline")))
    sample_loader = Mock()
    monkeypatch.setattr(workflow, "load_latest_sample_stock", sample_loader)
    answer = run_stock_advice_workflow("삼성전자 전망", replace(settings, stock_api_provider="toss"))
    assert "주가 조회 실패 (ConnectError)" in answer
    assert "주가·추세를 확보하지 못했습니다" in answer
    sample_loader.assert_not_called()


# 모의 API로 질문→일봉 저장→전처리 저장→현재가→컨텍스트→응답 전체 검증
def test_toss_workflow_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.parsers.stock_question_parser.load_stock_references",
        lambda: [{"stock_name": "삼성전자", "stock_code": "005930", "market": "KOSPI"}],
    )
    raw_path = tmp_path / "raw" / "005930_daily.json"
    trend_path = tmp_path / "processed" / "005930_trend.json"
    monkeypatch.setattr("src.preprocessing.stock_preprocessor.get_processed_price_trend_path", lambda code: trend_path)
    monkeypatch.setattr(workflow, "get_raw_price_path", lambda code: raw_path)
    config = replace(settings, stock_api_provider="toss", stock_api_url="https://example.test")
    manager = Mock()
    manager.get_access_token.return_value = "test-token"
    paths = []

    # 실제 증권사 통신 대신 캘린더·과거 일봉·현재가 제공
    def respond(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/market-calendar/KR"):
            result = {
                "today": {"date": "2026-10-20", "integrated": {"regularMarket": {
                    "startTime": "2026-10-20T09:00:00+09:00", "endTime": "2026-10-20T15:30:00+09:00",
                }}},
                "previousBusinessDay": {"date": "2026-10-19"},
            }
        elif request.url.path.endswith("/candles"):
            assert request.url.params["count"] == "4"
            assert request.url.params["before"][:10] == "2026-10-19"
            result = {"candles": [
                {"timestamp": f"{day}T00:00:00+09:00", "openPrice": "100", "highPrice": "120",
                 "lowPrice": "80", "closePrice": str(close), "volume": "1000", "currency": "KRW"}
                for day, close in zip(["2026-10-19", "2026-10-16", "2026-10-15", "2026-10-14"], [110, 105, 100, 95])
            ]}
        else:
            result = [{"symbol": "005930", "timestamp": "2026-10-20T12:00:00+09:00", "lastPrice": "115", "currency": "KRW"}]
        return httpx.Response(200, json={"result": result})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        # 기존 수집 함수를 모의 API와 임시 저장 경로에 연결
        def collect(stock_name, stock_code, **kwargs):
            kwargs.pop("http_client", None)
            return collect_stock_price(
                stock_name, stock_code, now=datetime.fromisoformat("2026-10-20T12:00:00+09:00"),
                token_manager=manager, http_client=client, file_path=raw_path, **kwargs,
            )

        monkeypatch.setattr(workflow, "collect_stock_price", collect)
        answer = run_stock_advice_workflow("삼성전자 최근 3거래일 사도 될까?", config)
    assert "현재가는 115원" in answer
    assert "최근 3거래일 상승" in answer
    assert "샘플 뉴스 흐름은" not in answer
    assert "확인된 관련 뉴스는 없습니다." in answer
    assert "등락률" not in answer and "%" not in answer
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 3
    trend = json.loads(trend_path.read_text(encoding="utf-8"))[-1]
    assert trend["base_date"] == "2026-10-19"
    assert trend["recent_days"] == 3
    assert paths == ["/api/v1/market-calendar/KR", "/api/v1/candles", "/api/v1/prices"]


# 질문에서 추출된 두 종목을 모두 조회하고 상대적 강세를 실제 상승으로 오인하지 않음
def test_multiple_stocks_are_passed_to_collectors(monkeypatch):
    parsed = ParsedQuestion(
        "삼전이 하이닉스보다 오르는 이유는?",
        (ParsedStock("삼전", "삼성전자", "005930", "target"),
         ParsedStock("하이닉스", "SK하이닉스", "000660", "comparison")),
        "comparison_reason",
    )
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    prices = {
        stock.stock_code: DailyStock(stock.stock_name, stock.stock_code, "2026-10-23",
                                    100, 110, 90, 95, 1000, -5.0, -5)
        for stock in parsed.stocks
    }
    trends = {
        stock.stock_code: StockTrend(stock.stock_name, stock.stock_code, "2026-10-23",
                                    5, -1.0 if index == 0 else -2.0, 1, 4, "최근 5거래일 하락")
        for index, stock in enumerate(parsed.stocks)
    }
    collector = Mock(side_effect=lambda name, code, **kwargs: prices[code])
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    monkeypatch.setattr(workflow, "load_latest_stock_trend", lambda code, *args, **kwargs: trends[code])
    answer = run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert collector.call_count == 2
    assert [call.args[1] for call in collector.call_args_list] == ["005930", "000660"]
    assert "삼성전자" in answer and "SK하이닉스" in answer
    assert "삼성전자의 가격 흐름이 상대적으로 더 강했습니다" in answer
    assert "인과관계는 추가 확인" in answer
    assert "%" not in answer


# 불명확한 종목과 미지원 날짜 범위는 주가 조회 전에 확인 메시지 반환
@pytest.mark.parametrize("period,clarification,expected", [
    (QueryPeriod(), True, "정확한 종목명을 알려주세요."),
    (QueryPeriod("date_range", None, "2026-09-26", "2026-10-26"), False, "최근 몇 거래일"),
])
def test_query_clarification_stops_execution(monkeypatch, period, clarification, expected):
    parsed = ParsedQuestion(
        "질문", (ParsedStock("삼전", "삼성전자", "005930"),),
        "general_analysis", period, clarification, "정확한 종목명을 알려주세요.",
    )
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    collector = Mock()
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    assert expected in run_stock_advice_workflow("질문", replace(settings, stock_api_provider="toss"))
    collector.assert_not_called()


# 현재 의도들은 주가·뉴스가 모두 필요하며 비교 의도는 두 종목을 같은 조건으로 조회
@pytest.mark.parametrize("intent", ["buy_opinion", "rise_reason", "decline_reason", "comparison_reason", "general_analysis"])
def test_intents_connect_actual_news_and_response(monkeypatch, saved_daily_prices_and_news, intent):
    stocks = (ParsedStock("삼성전자", "삼성전자", "005930"),)
    if intent == "comparison_reason":
        stocks += (ParsedStock("SK하이닉스", "SK하이닉스", "000660", "comparison"),)
    parsed = ParsedQuestion("분석 질문", stocks, intent)
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    collector = Mock(side_effect=lambda name, code, **kwargs: DailyStock(
        name, code, "2026-10-23", 100, 120, 90, 110, 1000, 10, 10,
    ))
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    monkeypatch.setattr(workflow, "load_latest_stock_trend", lambda code, *args, **kwargs: StockTrend(
        next(stock.stock_name for stock in stocks if stock.stock_code == code), code,
        "2026-10-23", 5, 5, 3, 2, "최근 5거래일 상승",
    ))

    # 같은 기사에 속한 두 청크를 원래 검색 순위대로 제공
    def search(question, **options):
        return [{"chunk_id": f"{options['stock_code']}_{i}", "news_id": options["stock_code"],
                 "stock_code": options["stock_code"], "title": "실적 기사", "content": f"근거 본문 {i}",
                 "url": f"https://example.test/{options['stock_code']}", "publisher": "경제신문",
                 "published_date": "2026-10-21", "chunk_index": i, "important": "O",
                 "vector_score": 0.8, "keyword_score": 2, "vector_normalized_score": 1,
                 "keyword_normalized_score": 1, "hybrid_score": 1} for i in (1, 2)]

    saved_daily_prices_and_news.side_effect = search
    sample = Mock(side_effect=AssertionError("Actual workflow must not use sample news"))
    monkeypatch.setattr(workflow, "load_recent_sample_news", sample)
    generator = Mock(wraps=workflow.generate_stock_advice)
    comparison = Mock(wraps=workflow.generate_stock_comparison_advice)
    monkeypatch.setattr(workflow, "generate_stock_advice", generator)
    monkeypatch.setattr(workflow, "generate_stock_comparison_advice", comparison)
    answer = run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert collector.call_count == saved_daily_prices_and_news.call_count == len(stocks)
    for call in saved_daily_prices_and_news.call_args_list:
        assert call.args[0] == parsed.original_question
        assert call.kwargs["start_date"] == "2026-10-19"
        assert call.kwargs["end_date"] == "2026-10-23"
    contexts = comparison.call_args.args[0] if len(stocks) > 1 else [generator.call_args.args[0]]
    for context in contexts:
        assert context["summary"] == {"news_count": 1, "news_chunk_count": 2}
        assert context["news"][0]["hybrid_score"] == 1
        assert context["data_sources"]["news"] == "hybrid_search"
    assert "근거 본문 1" in answer and "근거 본문 2" in answer
    assert "경제신문" in answer and "https://example.test/" in answer
    assert "샘플 뉴스" not in answer
    if intent == "rise_reason":
        assert "상승은 확인되지만" in answer
    sample.assert_not_called()


# 실제 인덱스 오류를 샘플로 대체하거나 자동 수집하지 않음
def test_hybrid_error_is_not_hidden(monkeypatch, saved_daily_prices_and_news):
    parsed = ParsedQuestion("질문", (ParsedStock("삼성전자", "삼성전자", "005930"),), "general_analysis")
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    monkeypatch.setattr(workflow, "collect_stock_price", Mock(return_value=DailyStock(
        "삼성전자", "005930", "2026-10-23", 100, 120, 90, 110, 1000, 10, 10,
    )))
    monkeypatch.setattr(workflow, "load_latest_stock_trend", Mock(return_value=StockTrend(
        "삼성전자", "005930", "2026-10-23", 5, 5, 3, 2, "최근 5거래일 상승",
    )))
    saved_daily_prices_and_news.side_effect = FileNotFoundError("Missing index")
    sample = Mock()
    monkeypatch.setattr(workflow, "load_recent_sample_news", sample)
    answer = run_stock_advice_workflow("질문", replace(settings, stock_api_provider="toss"))
    assert "뉴스 검색 실패 (FileNotFoundError)" in answer
    sample.assert_not_called()


# 날짜 차이 대신 완료 일봉의 거래일 순서로 기간 결정하며 주말·휴일을 제외하지 않음
def test_news_search_period_uses_saved_trading_days():
    assert workflow._get_news_search_period("005930", 3, "2026-10-19", "2026-10-20") == (
        "2026-10-15", "2026-10-20",
    )


# 부족하거나 기준일이 맞지 않는 저장 데이터로 임의 기간을 만들지 않음
@pytest.mark.parametrize("days, completed, end", [
    (20, "2026-10-23", "2026-10-23"), (5, "2026-10-24", "2026-10-24"),
    (5, "2026-10-23", "2026-10-22"),
])
def test_news_search_period_rejects_insufficient_data(days, completed, end):
    with pytest.raises(ValueError, match="insufficient"):
        workflow._get_news_search_period("005930", days, completed, end)


# 발굴 전 Agent가 종료를 선택하면 종목별 조회·빈 비교 응답을 생성하지 않음
def test_discovery_stop_before_execution(monkeypatch, saved_daily_prices_and_news):
    parsed = ParsedQuestion("반도체 소재 관련주 3개 추천", (), "stock_discovery",
                            discovery=StockDiscoveryCriteria("반도체 소재", 3))
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    collector, agent = Mock(), Mock(return_value=AgentDecision("stop", "검증 가능한 뉴스 근거 없음"))
    monkeypatch.setattr(workflow, "collect_stock_price", collector)
    monkeypatch.setattr(workflow, "decide_next_action", agent)
    answer = run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert "검증된 분석 후보를 확보하지 못했습니다" in answer
    assert "검증 가능한 뉴스 근거 없음" in answer
    collector.assert_not_called()
    saved_daily_prices_and_news.assert_not_called()
    agent.assert_called_once()

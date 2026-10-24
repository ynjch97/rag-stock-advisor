from dataclasses import replace
from datetime import datetime
import json
from unittest.mock import Mock

import httpx
import pytest

from src.app import stock_advice_workflow as workflow
from src.app.stock_advice_workflow import run_stock_advice_workflow
from src.collectors.stock_collector import collect_stock_price
from src.config.settings import settings
from src.domain.stock_models import DailyStock, RealtimeStock, StockTrend
from src.parsers.stock_question_parser import ParsedQuestion


def test_run_stock_advice_workflow_generates_answer(monkeypatch):
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


# 실제 조회 분기의 현재가·종가·추세 전달과 등락률 비노출 검증
@pytest.mark.parametrize("realtime", [True, False])
def test_toss_workflow_passes_price_and_trend(monkeypatch, realtime):
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
    collector.assert_called_once_with("두산에너빌리티", "034020", app_settings=config, recent_days=3)
    loader.assert_called_once_with("034020", 3, base_date="2026-10-23")
    context = generator.call_args.args[0]
    assert context["stock"]["current_price"] == 58900
    assert context["stock"]["change_rate"] == -3.76
    assert context["stock_trend"]["recent_change_rate"] == -5.5
    assert context["question"]["recent_days"] == 3
    assert context["news"] == []
    assert context["data_sources"] == {"stock": "toss", "news": "sample"}
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
        "삼성전자 전망", "삼성전자", "005930", "general_analysis",
    )))
    monkeypatch.setattr(workflow, "collect_stock_price", Mock(side_effect=httpx.ConnectError("offline")))
    sample_loader = Mock()
    monkeypatch.setattr(workflow, "load_latest_sample_stock", sample_loader)
    with pytest.raises(httpx.ConnectError):
        run_stock_advice_workflow("삼성전자 전망", replace(settings, stock_api_provider="toss"))
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
            return collect_stock_price(
                stock_name, stock_code, now=datetime.fromisoformat("2026-10-20T12:00:00+09:00"),
                token_manager=manager, http_client=client, file_path=raw_path, **kwargs,
            )

        monkeypatch.setattr(workflow, "collect_stock_price", collect)
        answer = run_stock_advice_workflow("삼성전자 최근 3거래일 사도 될까?", config)
    assert "현재가는 115원" in answer
    assert "최근 3거래일 상승" in answer
    assert "샘플 뉴스 흐름은" in answer
    assert "등락률" not in answer and "%" not in answer
    assert len(json.loads(raw_path.read_text(encoding="utf-8"))) == 3
    trend = json.loads(trend_path.read_text(encoding="utf-8"))[-1]
    assert trend["base_date"] == "2026-10-19"
    assert trend["recent_days"] == 3
    assert paths == ["/api/v1/market-calendar/KR", "/api/v1/candles", "/api/v1/prices"]

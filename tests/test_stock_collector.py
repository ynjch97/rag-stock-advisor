from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import Mock

import httpx
import pytest

from src.collectors.stock_collector import collect_stock_price, determine_market_status, get_market_status
from src.config.settings import settings
from src.domain.stock_models import DailyStock, RealtimeStock


# 정상 거래일의 모의 시장 캘린더 생성
@pytest.fixture
def calendar():
    return {
        "today": {
            "date": "2026-10-20",
            "integrated": {"regularMarket": {
                "startTime": "2026-10-20T09:00:00+09:00",
                "endTime": "2026-10-20T15:30:00+09:00",
            }},
        },
        "previousBusinessDay": {"date": "2026-10-19"},
    }


# 장 시작·종료 경계 시각 및 당일 완료 거래일 선택 검증
@pytest.mark.parametrize("clock,status,base,completed,realtime", [
    ("08:59:59", "before_market", "2026-10-19", "2026-10-19", False),
    ("09:00:00", "regular_market", "2026-10-20", "2026-10-19", True),
    ("15:29:59", "regular_market", "2026-10-20", "2026-10-19", True),
    ("15:30:00", "after_market", "2026-10-20", "2026-10-20", False),
    ("21:00:00", "after_market", "2026-10-20", "2026-10-20", False),
])
def test_session_boundaries(calendar, clock, status, base, completed, realtime):
    result = determine_market_status(
        calendar, datetime.fromisoformat(f"2026-10-20T{clock}+09:00"),
    )
    assert (result.status, result.base_date) == (status, base)
    assert result.latest_completed_date == completed
    assert result.is_realtime is realtime


# 휴장일의 직전 거래일을 API 값 그대로 사용하며 주말 여부로 추정하지 않음
def test_holiday_uses_previous_business_day(calendar):
    calendar["today"]["integrated"] = None
    calendar["previousBusinessDay"]["date"] = "2026-10-16"
    result = determine_market_status(calendar, datetime.fromisoformat("2026-10-20T12:00:00+09:00"))
    assert result.status == "holiday"
    assert result.base_date == result.latest_completed_date == "2026-10-16"


# 특별 개장일에는 고정 시간이 아닌 API의 정규장 시간을 사용
def test_special_session_and_utc_time(calendar):
    calendar["today"]["integrated"]["regularMarket"]["startTime"] = "2026-10-20T10:00:00+09:00"
    result = determine_market_status(calendar, datetime.fromisoformat("2026-10-20T00:30:00+00:00"))
    assert result.status == "before_market"
    assert result.checked_at.hour == 9


# 잘못된 캘린더를 휴장일로 오인하지 않고 오류 처리
@pytest.mark.parametrize("case", ["missing", "date", "previous", "session", "order", "naive"])
def test_invalid_calendar_is_rejected(calendar, case):
    if case == "missing":
        del calendar["today"]["integrated"]
    elif case == "date":
        calendar["today"]["date"] = "2026-10-19"
    elif case == "previous":
        calendar["previousBusinessDay"]["date"] = "2026-10-20"
    elif case == "session":
        calendar["today"]["integrated"]["regularMarket"] = None
    elif case == "order":
        calendar["today"]["integrated"]["regularMarket"]["endTime"] = "2026-10-20T08:00:00+09:00"
    else:
        calendar["today"]["integrated"]["regularMarket"]["startTime"] = "2026-10-20T09:00:00"
    with pytest.raises(ValueError):
        determine_market_status(calendar, datetime.fromisoformat("2026-10-20T12:00:00+09:00"))


# 타임존이 없는 현재 시각의 시스템 타임존 의존 방지
def test_naive_now_is_rejected(calendar):
    with pytest.raises(ValueError, match="timezone"):
        determine_market_status(calendar, datetime(2026, 10, 20, 12))


# provider 분기 시 sample은 HTTP 호출 및 토큰 발급 생략
def test_sample_and_unsupported_provider():
    token_manager = Mock()
    assert get_market_status(
        app_settings=replace(settings, stock_api_provider="sample"),
        token_manager=token_manager,
    ) is None
    token_manager.get_access_token.assert_not_called()
    with pytest.raises(ValueError, match="Unsupported"):
        get_market_status(app_settings=replace(settings, stock_api_provider="unknown"))


# 캘린더 API의 인증 헤더·한국 기준일 전달 및 HTTP 오류 전파 검증
@pytest.mark.parametrize("status_code", [200, 401, 429, 500])
def test_calendar_api(calendar, status_code):
    config = replace(settings, stock_api_provider="toss", stock_api_url="https://example.test")
    manager = Mock()
    manager.get_access_token.return_value = "test-token"

    # 인증 헤더와 기준일 쿼리 검사 후 모의 캘린더 응답
    def respond(request):
        assert request.url.path == "/api/v1/market-calendar/KR"
        assert request.url.params["date"] == "2026-10-20"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(status_code, json={"result": calendar})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        arguments = dict(
            now=datetime.fromisoformat("2026-10-20T12:00:00+09:00"),
            app_settings=config, token_manager=manager, http_client=client,
        )
        if status_code == 200:
            assert get_market_status(**arguments).status == "regular_market"
        else:
            with pytest.raises(httpx.HTTPStatusError):
                get_market_status(**arguments)


# 시장 캘린더·일봉·현재가 응답을 주입하여 조회 흐름 검증
@pytest.fixture
def price_api(calendar):
    config = replace(settings, stock_api_provider="toss", stock_api_url="https://example.test")
    manager = Mock()
    manager.get_access_token.return_value = "test-token"
    requests = []
    candle = {
        "openPrice": "100", "highPrice": "110", "lowPrice": "90",
        "closePrice": "100", "volume": "1000", "currency": "KRW",
    }
    quote = {
        "symbol": "005930", "timestamp": "2026-10-20T12:00:00+09:00",
        "lastPrice": "105", "currency": "KRW",
    }

    # 요청 파라미터 기준으로 일봉 날짜를 채우고 실제 통신 없이 응답
    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path.endswith("/market-calendar/KR"):
            return httpx.Response(200, json={"result": calendar})
        if request.url.path.endswith("/candles"):
            assert request.url.params["symbol"] == "005930"
            assert request.url.params["interval"] == "1d"
            assert request.url.params["count"] in ("1", "2")
            assert request.url.params["adjusted"] == "false"
            target = request.url.params["before"][:10]
            data = {"timestamp": f"{target}T00:00:00+09:00", **candle}
            candles = [data]
            if request.url.params["count"] == "2":
                previous_date = (datetime.fromisoformat(target) - timedelta(days=1)).date()
                if previous_date.weekday() == 6:
                    previous_date -= timedelta(days=2)
                candles.append({
                    "timestamp": candle.get("previousTimestamp", f"{previous_date}T00:00:00+09:00"),
                    "closePrice": candle.get("previousClose", "100"), "currency": "KRW",
                })
            return httpx.Response(200, json={"result": {"candles": candles}})
        assert request.url.path.endswith("/prices")
        assert request.url.params["symbols"] == "005930"
        return httpx.Response(200, json={"result": [quote]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        yield config, manager, client, requests, candle, quote


# 장중에는 직전 거래일 종가 대비 현재가 등락률 계산
@pytest.mark.parametrize("previous_close,expected_rate", [(100, 5.0), (110, -4.55), (105, 0.0)])
def test_collect_realtime_stock(price_api, previous_close, expected_rate):
    config, manager, client, requests, candle, _ = price_api
    candle["closePrice"] = str(previous_close)
    result = collect_stock_price(
        "삼성전자", "005930", datetime.fromisoformat("2026-10-20T12:00:00+09:00"),
        config, manager, client,
    )
    assert result == RealtimeStock("삼성전자", "005930", 105, expected_rate, "2026-10-20")
    assert requests[1].url.params["before"] == "2026-10-19T23:59:59+09:00"
    assert not client.is_closed


# 장 전·장 후·휴장일의 기준일 일봉 선택 및 현재가 호출 생략 검증
@pytest.mark.parametrize("clock,holiday,target", [
    ("08:00:00", False, "2026-10-19"),
    ("15:30:00", False, "2026-10-20"),
    ("12:00:00", True, "2026-10-19"),
])
def test_collect_daily_stock(price_api, calendar, clock, holiday, target):
    config, manager, client, requests, _, _ = price_api
    if holiday:
        calendar["today"]["integrated"] = None
    result = collect_stock_price(
        "삼성전자", "005930", datetime.fromisoformat(f"2026-10-20T{clock}+09:00"),
        config, manager, client,
    )
    assert result == DailyStock("삼성전자", "005930", target, 100, 110, 90, 100, 1000, 0.0, 0)
    assert result.base_date == target
    assert len(requests) == 2
    assert requests[1].url.params["before"] == f"{target}T23:59:59+09:00"


# 전일 종가 대비 상승·하락·보합 및 소수점 반올림 검증
@pytest.mark.parametrize("previous_close,close,rate,price", [
    (100, 110, 10.0, 10), (110, 100, -9.09, -10), (100, 100, 0.0, 0),
])
def test_daily_stock_changes(price_api, previous_close, close, rate, price):
    config, manager, client, _, candle, _ = price_api
    candle.update(openPrice="95", closePrice=str(close), previousClose=str(previous_close))
    result = collect_stock_price(
        "삼성전자", "005930", datetime.fromisoformat("2026-10-20T16:00:00+09:00"),
        config, manager, client,
    )
    assert result.change_rate == rate
    assert result.change_price == price


# 전일 종가가 0이거나 동일 거래일 캔들이면 계산하지 않음
@pytest.mark.parametrize("field,value", [
    ("previousClose", "0"),
    ("previousTimestamp", "2026-10-20T00:00:00+09:00"),
])
def test_invalid_previous_candle(price_api, field, value):
    config, manager, client, _, candle, _ = price_api
    candle[field] = value
    with pytest.raises(ValueError):
        collect_stock_price(
            "삼성전자", "005930", datetime.fromisoformat("2026-10-20T16:00:00+09:00"),
            config, manager, client,
        )


# 잘못된 기준일·가격·통화 응답을 정상 주가로 반환하지 않음
@pytest.mark.parametrize("field,value", [
    ("timestamp", "2026-10-16T00:00:00+09:00"),
    ("closePrice", "0"), ("closePrice", "NaN"), ("closePrice", "100.5"),
    ("highPrice", "80"), ("volume", "-1"), ("currency", "USD"),
])
def test_invalid_daily_price_is_rejected(price_api, field, value):
    config, manager, client, _, candle, _ = price_api
    candle[field] = value
    with pytest.raises(ValueError):
        collect_stock_price(
            "삼성전자", "005930", datetime.fromisoformat("2026-10-20T16:00:00+09:00"),
            config, manager, client,
        )


# 전일의 현재가나 다른 종목 응답을 현재 실시간 데이터로 오인하지 않음
@pytest.mark.parametrize("field,value", [
    ("symbol", "000660"), ("currency", "USD"),
    ("timestamp", "2026-10-19T19:59:59+09:00"), ("lastPrice", "-1"),
])
def test_invalid_realtime_price_is_rejected(price_api, field, value):
    config, manager, client, _, _, quote = price_api
    quote[field] = value
    with pytest.raises(ValueError):
        collect_stock_price(
            "삼성전자", "005930", datetime.fromisoformat("2026-10-20T12:00:00+09:00"),
            config, manager, client,
        )


# sample provider는 API 없이 기존 주가를 모델로 변환하고 코드 불일치 검증
def test_collect_sample_stock():
    config = replace(settings, stock_api_provider="sample")
    result = collect_stock_price("삼성전자", "005930", app_settings=config)
    assert isinstance(result, RealtimeStock)
    assert result.stock_code == "005930"
    with pytest.raises(ValueError, match="does not match"):
        collect_stock_price("삼성전자", "000660", app_settings=config)
    with pytest.raises(ValueError, match="Unsupported"):
        collect_stock_price("삼성전자", "005930", app_settings=replace(config, stock_api_provider="other"))

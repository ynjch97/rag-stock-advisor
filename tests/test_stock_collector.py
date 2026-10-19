from dataclasses import replace
from datetime import datetime
from unittest.mock import Mock

import httpx
import pytest

from src.collectors.stock_collector import determine_market_status, get_market_status
from src.config.settings import settings


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

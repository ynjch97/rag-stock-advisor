from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from src.collectors.stock_token_manager import StockTokenManager, stock_token_manager
from src.config.settings import Settings, settings
from src.domain.stock_models import MarketStatus


# 한국 시장의 날짜와 시각 비교 기준 (UTC+09:00)
KST = timezone(timedelta(hours=9))
MARKET_CALENDAR_TIMEOUT_SECONDS = 10.0


# 토스 시장 캘린더로 현재 상태 조회; sample 모드는 캘린더 조회 생략
def get_market_status(
    now: datetime | None = None,
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
) -> MarketStatus | None:
    provider = (app_settings.stock_api_provider or "").lower()
    if provider == "sample":
        return None
    if provider == "toss":
        checked_at = _normalize_market_time(now) # 조회 시점의 현재 한국 시각
        if not app_settings.stock_api_url:
            raise ValueError("Toss API requires STOCK_API_URL.")
        if token_manager is None:
            token_manager = (
                stock_token_manager if app_settings is settings
                else StockTokenManager(app_settings, http_client=http_client)
            )
        token = token_manager.get_access_token()
        if not token:
            raise ValueError("Toss API requires an access token.")
        if http_client is not None:
            calendar = _request_market_calendar(
                http_client, app_settings.stock_api_url, token, checked_at.date(),
            )
        else:
            with httpx.Client(timeout=MARKET_CALENDAR_TIMEOUT_SECONDS) as client:
                calendar = _request_market_calendar(
                    client, app_settings.stock_api_url, token, checked_at.date(),
                )
        return determine_market_status(calendar, checked_at)
    raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")


# 조회 기준일을 지정하여 캘린더의 result 객체 반환; HTTP 오류는 호출자에게 전달
def _request_market_calendar(
    client: httpx.Client, base_url: str, token: str, target_date: date,
) -> dict[str, Any]:
    response = client.get(
        f"{base_url.rstrip('/')}/api/v1/market-calendar/KR",
        params={"date": target_date.isoformat()},
        headers={"Authorization": f"Bearer {token}"},
        timeout=MARKET_CALENDAR_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise ValueError("Invalid market calendar response: result must be an object.")
    return payload["result"]


# 캘린더 result와 한국 시각을 비교하여 상태·조회일·마지막 완료 거래일 결정
def determine_market_status(
    calendar: dict[str, Any], now: datetime | None = None,
) -> MarketStatus:
    checked_at = _normalize_market_time(now)
    try:
        today = calendar["today"]
        today_date = date.fromisoformat(today["date"])
        previous_date = date.fromisoformat(calendar["previousBusinessDay"]["date"])
        integrated = today["integrated"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid market calendar dates or missing integrated field.") from None

    if today_date != checked_at.date() or previous_date >= today_date:
        raise ValueError("Market calendar dates do not match the requested date.")

    if integrated is None:
        return MarketStatus(
            status="holiday", base_date=previous_date.isoformat(),
            latest_completed_date=previous_date.isoformat(), is_realtime=False,
            checked_at=checked_at,
        )

    try:
        regular_market = integrated["regularMarket"]
        start = _parse_session_time(regular_market["startTime"])
        end = _parse_session_time(regular_market["endTime"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid or missing regular market session.") from None
    if start.date() != today_date or end.date() != today_date or start >= end:
        raise ValueError("Invalid regular market session dates or time order.")

    if checked_at < start:
        status = "before_market"
        base_date = previous_date
    elif checked_at < end:
        status = "regular_market"
        base_date = today_date
    else:
        status = "after_market"
        base_date = today_date

    completed_date = today_date if status == "after_market" else previous_date
    return MarketStatus(
        status=status, base_date=base_date.isoformat(),
        latest_completed_date=completed_date.isoformat(),
        is_realtime=status == "regular_market", checked_at=checked_at,
    )


# 시스템 시각은 한국 시간으로 생성하고 주입된 시각은 타임존 포함 여부 검증
def _normalize_market_time(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(KST)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Market time must include a timezone offset.")
    return now.astimezone(KST)


# API의 ISO 8601 세션 시각을 한국 시간으로 변환
def _parse_session_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Session time must be an ISO 8601 string.")
    return _normalize_market_time(datetime.fromisoformat(value.replace("Z", "+00:00")))

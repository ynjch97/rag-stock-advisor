from datetime import date, datetime, timedelta, timezone
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from src.collectors.stock_token_manager import StockTokenManager, stock_token_manager
from src.config.settings import Settings, settings
from src.data_loaders.sample_stock_loader import load_latest_sample_stock
from src.domain.stock_models import DailyStock, MarketStatus, RealtimeStock


# 한국 시장의 날짜와 시각 비교 기준 (UTC+09:00)
KST = timezone(timedelta(hours=9))
MARKET_CALENDAR_TIMEOUT_SECONDS = 10.0
STOCK_REQUEST_TIMEOUT_SECONDS = 10.0


# 시장 상태에 따라 현재가 또는 기준일 일봉 조회
def collect_stock_price(
    stock_name: str,
    stock_code: str,
    now: datetime | None = None,
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
) -> DailyStock | RealtimeStock:
    if not stock_name.strip() or not stock_code.strip():
        raise ValueError("Stock name and stock code must not be empty.")
    provider = (app_settings.stock_api_provider or "").lower()
    if provider == "sample":
        stock = load_latest_sample_stock(stock_name)
        if stock["stock_code"] != stock_code:
            raise ValueError("Sample stock code does not match the requested stock.")
        return RealtimeStock(
            stock_name=stock["stock_name"], stock_code=stock["stock_code"],
            current_price=stock["current_price"], change_rate=stock["change_rate"],
            base_date=stock["base_date"],
        )
    if provider != "toss":
        raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")
    if not app_settings.stock_api_url:
        raise ValueError("Toss API requires STOCK_API_URL.")
    if token_manager is None:
        token_manager = (
            stock_token_manager if app_settings is settings
            else StockTokenManager(app_settings, http_client=http_client)
        )
    # 주입된 클라이언트는 유지하고 내부에서 생성한 클라이언트만 닫음
    with (
        nullcontext(http_client) if http_client is not None
        else httpx.Client(timeout=STOCK_REQUEST_TIMEOUT_SECONDS)
    ) as client:
        market = get_market_status(now, app_settings, token_manager, client)
        if market is None:
            raise ValueError("Toss market status is required.")
        token = token_manager.get_access_token()
        if not token:
            raise ValueError("Toss API requires an access token.")
        base_url = app_settings.stock_api_url.rstrip("/")
        # 장중이 아닌 경우 (장 전후, 휴장일)
        if not market.is_realtime:
            return _get_daily_stock(client, base_url, token, stock_name, stock_code, market.base_date)

        # 장중인 경우 (직전 거래일 종가 확보 후 현재가 조회)
        previous_candles = _request_daily_candles(
            client, base_url, token, stock_code, market.latest_completed_date, count=1,
        )
        previous_close = _get_candle_close(previous_candles[0], market.latest_completed_date)
        response = client.get(
            f"{base_url}/api/v1/prices", params={"symbols": stock_code},
            headers={"Authorization": f"Bearer {token}"}, timeout=STOCK_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        quotes = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
            raise ValueError("Expected one realtime stock quote.")
        quote = quotes[0]
        if quote.get("symbol") != stock_code or quote.get("currency") != "KRW":
            raise ValueError("Realtime stock symbol or currency mismatch.")
        timestamp = _parse_session_time(quote.get("timestamp"))
        if timestamp.date().isoformat() != market.base_date:
            raise ValueError("Realtime stock date does not match the requested date.")
        price = _parse_stock_integer(quote.get("lastPrice"))
        change_rate = _calculate_change_rate(price, previous_close)
        return RealtimeStock(
            stock_name=stock_name, stock_code=stock_code, current_price=price,
            change_rate=change_rate, base_date=market.base_date,
        )


# 기준일과 직전 거래일 일봉으로 OHLCV 및 전일 종가 대비 등락 정보 생성
def _get_daily_stock(
    client: httpx.Client, base_url: str, token: str,
    stock_name: str, stock_code: str, base_date: str,
) -> DailyStock:
    candles = _request_daily_candles(client, base_url, token, stock_code, base_date, count=2)
    candle, previous = candles
    close = _get_candle_close(candle, base_date)
    previous_date = _parse_session_time(previous.get("timestamp")).date()
    if previous_date >= date.fromisoformat(base_date):
        raise ValueError("Previous candle must precede the requested date.")
    previous_close = _get_candle_close(previous, previous_date.isoformat())
    opening, high, low = (
        _parse_stock_integer(candle.get(field))
        for field in ("openPrice", "highPrice", "lowPrice")
    )
    volume = _parse_stock_integer(candle.get("volume"), allow_zero=True)
    if not low <= min(opening, close) <= max(opening, close) <= high:
        raise ValueError("Invalid daily stock OHLC price range.")
    return DailyStock(
        stock_name=stock_name, stock_code=stock_code, base_date=base_date,
        open=opening, high=high, low=low, close=close, volume=volume,
        change_rate=_calculate_change_rate(close, previous_close),
        change_price=close - previous_close,
    )


# 기준일 이하의 최신 일봉을 지정 개수만큼 조회하며 부족한 응답은 오류 처리
def _request_daily_candles(
    client: httpx.Client, base_url: str, token: str,
    stock_code: str, base_date: str, count: int,
) -> list[dict[str, Any]]:
    target_date = date.fromisoformat(base_date)
    response = client.get(
        f"{base_url}/api/v1/candles",
        params={
            "symbol": stock_code, "interval": "1d", "count": count,
            "before": f"{target_date.isoformat()}T23:59:59+09:00",
            "adjusted": "false",
        },
        headers={"Authorization": f"Bearer {token}"}, timeout=STOCK_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result") if isinstance(payload, dict) else None
    candles = result.get("candles") if isinstance(result, dict) else None
    if (
        not isinstance(candles, list) or len(candles) != count
        or any(not isinstance(candle, dict) for candle in candles)
    ):
        raise ValueError(f"Expected {count} daily stock candle(s), including previous close.")
    return candles


# 캔들의 기준일·통화를 확인하고 양수 종가 반환
def _get_candle_close(candle: dict[str, Any], base_date: str) -> int:
    timestamp = _parse_session_time(candle.get("timestamp"))
    if timestamp.date().isoformat() != base_date:
        raise ValueError("Daily stock date does not match the requested date.")
    if candle.get("currency") != "KRW":
        raise ValueError("Daily stock currency must be KRW.")
    return _parse_stock_integer(candle.get("closePrice"))


# 직전 거래일 종가 대비 변동률을 소수점 둘째 자리까지 계산
def _calculate_change_rate(price: int, previous_close: int) -> float:
    return float(round(Decimal(price - previous_close) / Decimal(previous_close) * 100, 2))


# 원화 가격·거래량을 정수로 변환하고 소수·음수·비정상 숫자 거부
def _parse_stock_integer(value: Any, allow_zero: bool = False) -> int:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Invalid stock price or volume.") from None
    if (
        not number.is_finite() or number != number.to_integral_value()
        or number < 0 or (number == 0 and not allow_zero)
    ):
        raise ValueError("Invalid stock price or volume.")
    return int(number)


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

from datetime import date, datetime, timedelta, timezone
from contextlib import nullcontext
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

import httpx

from src.collectors.stock_token_manager import StockTokenManager, stock_token_manager
from src.config.paths import get_raw_price_path
from src.config.settings import Settings, settings
from src.data_loaders.sample_stock_loader import load_latest_sample_stock
from src.domain.stock_models import DEFAULT_RECENT_DAYS, DailyStock, MarketStatus, RealtimeStock
from src.preprocessing.stock_preprocessor import preprocess_and_save_stock_trend


# 한국 시장의 날짜와 시각 비교 기준 (UTC+09:00)
KST = timezone(timedelta(hours=9))
MARKET_CALENDAR_TIMEOUT_SECONDS = 10.0
STOCK_REQUEST_TIMEOUT_SECONDS = 10.0


# 완료된 최근 n거래일을 수집·병합·저장하고 (최근 일봉 목록, 변경 여부) 반환
def collect_and_save_daily_stocks(
    stock_name: str,
    stock_code: str,
    now: datetime | None = None,
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
    file_path: Path | None = None,
    *,
    market_status: MarketStatus | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> tuple[list[DailyStock], bool]:
    if type(recent_days) is not int or not 1 <= recent_days <= 199:
        raise ValueError("recent_days must be between 1 and 199 trading days.")
    if not stock_name.strip() or re.fullmatch(r"[A-Za-z0-9]{6}", stock_code) is None:
        raise ValueError("A stock name and a six-character domestic stock code are required.")
    provider = (app_settings.stock_api_provider or "").lower()
    if provider == "sample":
        # 기존 샘플에는 OHLCV가 없으므로 raw 일봉을 생성하지 않음
        return [], False
    if provider != "toss":
        raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")
    if not app_settings.stock_api_url:
        raise ValueError("Toss API requires STOCK_API_URL.")
    destination = file_path if file_path is not None else get_raw_price_path(stock_code)
    stored = _load_saved_daily_stocks(destination, stock_code)
    if token_manager is None:
        token_manager = (
            stock_token_manager if app_settings is settings
            else StockTokenManager(app_settings, http_client=http_client)
        )
    with (
        nullcontext(http_client) if http_client is not None
        else httpx.Client(timeout=STOCK_REQUEST_TIMEOUT_SECONDS)
    ) as client:
        market = market_status if market_status is not None else get_market_status(
            now, app_settings, token_manager, client,
        )
        if market is None:
            raise ValueError("Toss market status is required.")
        token = token_manager.get_access_token()
        if not token:
            raise ValueError("Toss API requires an access token.")
        candles = _request_daily_candles(
            client, app_settings.stock_api_url.rstrip("/"), token,
            stock_code, market.latest_completed_date, recent_days + 1,
        )
        dates = [_parse_session_time(candle.get("timestamp")).date() for candle in candles]
        if dates[0].isoformat() != market.latest_completed_date or any(
            newer <= older for newer, older in zip(dates, dates[1:])
        ):
            raise ValueError("Daily candles must start on the completed date in descending order.")
        recent = [
            _build_daily_stock(stock_name, stock_code, day.isoformat(), candle, previous)
            for day, candle, previous in zip(dates, candles, candles[1:])
        ]

    merged = {stock.base_date: stock for stock in stored}
    merged.update({stock.base_date: stock for stock in recent})
    all_stocks = [merged[day] for day in sorted(merged)]
    changed = stored != all_stocks
    if changed:
        _save_daily_stocks(destination, all_stocks)
    return sorted(recent, key=lambda stock: stock.base_date), changed


# 기존 JSON 일봉을 읽고 필수 필드·종목코드·거래일·숫자 값 검증
def _load_saved_daily_stocks(file_path: Path, stock_code: str) -> list[DailyStock]:
    try:
        with file_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
    except FileNotFoundError:
        return []
    if not isinstance(records, list):
        raise ValueError("Saved daily stocks must contain a JSON array.")
    stocks = []
    for record in records:
        try:
            stock = DailyStock(**record)
            valid = (
                stock.stock_code == stock_code and isinstance(stock.stock_name, str)
                and bool(stock.stock_name.strip())
                and date.fromisoformat(stock.base_date).isoformat() == stock.base_date
                and all(type(value) is int and value > 0 for value in (
                    stock.open, stock.high, stock.low, stock.close,
                ))
                and type(stock.volume) is int and stock.volume >= 0
                and stock.low <= min(stock.open, stock.close) <= max(stock.open, stock.close) <= stock.high
                and type(stock.change_price) is int
                and type(stock.change_rate) in (float, int) and math.isfinite(stock.change_rate)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Invalid saved daily stock record.")
        stocks.append(stock)
    return stocks


# 임시 파일에 전체 데이터를 쓴 뒤 교체하여 저장 실패 시 기존 파일 보존
def _save_daily_stocks(file_path: Path, stocks: list[DailyStock]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=file_path.parent, suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump([asdict(stock) for stock in stocks], file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# 완료 일봉·추세를 저장하고 장중에는 현재가, 장 외에는 기준일 일봉 반환
def collect_stock_price(
    stock_name: str,
    stock_code: str,
    now: datetime | None = None,
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
    file_path: Path | None = None,
    *,
    recent_days: int = DEFAULT_RECENT_DAYS,
    trend_file_path: Path | None = None,
) -> DailyStock | RealtimeStock:
    if type(recent_days) is not int or not 1 <= recent_days <= 199:
        raise ValueError("recent_days must be between 1 and 199 trading days.")
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
        # 장 후에는 당일, 장 전/장중/휴장일에는 직전 거래일부터 완료 일봉 n개 수집·저장
        stocks, data_changed = collect_and_save_daily_stocks(
            stock_name, stock_code, now=now, app_settings=app_settings,
            token_manager=token_manager, http_client=client, file_path=file_path,
            market_status=market, recent_days=recent_days,
        )
        preprocess_and_save_stock_trend(
            stocks, recent_days, trend_file_path, data_changed=data_changed,
        )
        # 장중이 아닌 경우 (장 전후, 휴장일): 기준일의 마감 일봉 반환
        if not market.is_realtime:
            return stocks[-1]

        token = token_manager.get_access_token()
        if not token:
            raise ValueError("Toss API requires an access token.")
        base_url = app_settings.stock_api_url.rstrip("/")
        # 장중인 경우: 확보한 직전 거래일 종가를 재사용하고 현재가만 추가 조회
        previous_close = stocks[-1].close
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


# 인접한 두 거래일 캔들을 검증하고 전일 종가 대비 등락 정보를 가진 일봉 생성
def _build_daily_stock(
    stock_name: str, stock_code: str, base_date: str,
    candle: dict[str, Any], previous: dict[str, Any],
) -> DailyStock:
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

import json
import math
import random
import tempfile
import time
from pathlib import Path

import httpx

from src.collectors.stock_token_manager import StockTokenManager, stock_token_manager
from src.config.paths import STOCK_NAME_CODE_PATH
from src.config.settings import Settings, settings
from src.data_loaders.sample_stock_loader import load_sample_stocks


STOCK_MARKETS = ("KOSPI", "KOSDAQ")
# 종목 목록 API의 네트워크 작업별 제한 시간(초)
REFERENCE_REQUEST_TIMEOUT_SECONDS = 30.0
# 전체 종목 조회의 초당 1회 제한을 위한 시장 간 대기 시간
REFERENCE_REQUEST_INTERVAL_SECONDS = 1.1
# 최초 요청 외에 허용하는 429 재시도 횟수
REFERENCE_MAX_RETRIES = 3


# provider별 종목 목록 수집; 샘플의 시장은 SAMPLE로 표시
def collect_stock_references(
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
) -> list[dict[str, str]]:
    provider = (app_settings.stock_api_provider or "").lower()
    if provider == "sample":
        pairs = {
            (stock["stock_code"], stock["stock_name"])
            for stock in load_sample_stocks()
        }
        return [
            {"stock_name": name, "stock_code": code, "market": "SAMPLE"}
            for code, name in sorted(pairs)
        ]
    if provider == "toss":
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
            return _collect_toss_references(http_client, app_settings.stock_api_url, token)
        with httpx.Client(timeout=REFERENCE_REQUEST_TIMEOUT_SECONDS) as client:
            return _collect_toss_references(client, app_settings.stock_api_url, token)
    raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")


# KOSPI·KOSDAQ 응답을 프로젝트 필드로 변환하여 하나의 목록으로 결합
def _collect_toss_references(
    client: httpx.Client, base_url: str, token: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    codes: set[str] = set()
    for index, market in enumerate(STOCK_MARKETS):
        if index:
            time.sleep(REFERENCE_REQUEST_INTERVAL_SECONDS)
        response = _request_stock_list(client, base_url, token, market)
        payload = response.json()
        stocks = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(stocks, list) or not stocks:
            raise ValueError(f"Invalid or empty stock list for {market}.")
        for stock in stocks:
            if not isinstance(stock, dict):
                raise ValueError(f"Invalid stock record for {market}.")
            name, code = stock.get("name"), stock.get("symbol")
            if any(not isinstance(value, str) or not value.strip() for value in (name, code)):
                raise ValueError(f"Missing stock name or symbol for {market}.")
            if code in codes:
                raise ValueError(f"Duplicate stock code: {code}")
            codes.add(code)
            records.append({"stock_name": name, "stock_code": code, "market": market})
    return sorted(records, key=lambda record: record["stock_code"])


# 429 응답에 한해 서버 권장 시간 또는 지수 백오프 후 최대 3회 재시도
def _request_stock_list(
    client: httpx.Client, base_url: str, token: str, market: str,
) -> httpx.Response:
    for attempt in range(REFERENCE_MAX_RETRIES + 1):
        response = client.get(
            f"{base_url.rstrip('/')}/api/v1/stocks/all",
            params={"market": market},
            headers={"Authorization": f"Bearer {token}"},
            timeout=REFERENCE_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 429 or attempt == REFERENCE_MAX_RETRIES:
            response.raise_for_status()
            return response
        delay = _get_retry_delay(response, attempt)
        # 긴 서버 권장 대기 시간은 호출자에게 오류로 전달하고 즉시 재요청하지 않음
        if delay > 60:
            response.raise_for_status()
        time.sleep(delay)
    raise RuntimeError("Stock list retry limit exceeded.")


# Retry-After의 초 값을 우선 사용하고 잘못된 값은 지수 백오프로 대체
def _get_retry_delay(response: httpx.Response, attempt: int) -> float:
    try:
        delay = float(response.headers.get("Retry-After", ""))
    except ValueError:
        delay = float(2 ** attempt)
    if not math.isfinite(delay) or delay < 0:
        delay = float(2 ** attempt)
    return max(REFERENCE_REQUEST_INTERVAL_SECONDS, delay) + random.uniform(0.1, 0.3)


# 전체 수집 성공 후 임시 파일을 교체하여 최초 저장 또는 기존 파일 갱신
def refresh_stock_references(
    file_path: Path = STOCK_NAME_CODE_PATH,
    app_settings: Settings = settings,
    token_manager: StockTokenManager | None = None,
    http_client: httpx.Client | None = None,
) -> list[dict[str, str]]:
    records = collect_stock_references(app_settings, token_manager, http_client)
    if not records:
        raise ValueError("Cannot save an empty stock reference list.")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=file_path.parent,
            suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return records

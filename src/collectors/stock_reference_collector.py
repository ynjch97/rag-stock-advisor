import json
import tempfile
from pathlib import Path

import httpx

from src.collectors.stock_token_manager import StockTokenManager, stock_token_manager
from src.config.paths import STOCK_NAME_CODE_PATH
from src.config.settings import Settings, settings
from src.data_loaders.sample_stock_loader import load_sample_stocks


STOCK_MARKETS = ("KOSPI", "KOSDAQ")
# 종목 목록 API의 네트워크 작업별 제한 시간(초)
REFERENCE_REQUEST_TIMEOUT_SECONDS = 30.0


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
    for market in STOCK_MARKETS:
        response = client.get(
            f"{base_url.rstrip('/')}/api/v1/stocks/all",
            params={"market": market},
            headers={"Authorization": f"Bearer {token}"},
            timeout=REFERENCE_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
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

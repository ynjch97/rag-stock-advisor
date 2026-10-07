import threading
import time
from typing import Any

import httpx

from src.config.settings import Settings, settings


# 토큰 만료 5분 전에 재발급하기 위한 여유 시간(초)
TOKEN_REFRESH_BUFFER_SECONDS = 300

# API 응답에 유효기간이 없거나 잘못된 경우 적용하는 기본 유효기간 24시간(초)
DEFAULT_TOKEN_TTL_SECONDS = 86400

# 토큰 발급 API 응답을 기다리는 최대 시간(초)
TOKEN_REQUEST_TIMEOUT_SECONDS = 10.0


# 주가 API 토큰 발급 과정에서 발생한 오류
class StockTokenError(RuntimeError):
    pass


# API 제공자에 따라 액세스 토큰을 발급하고 메모리에 캐시
class StockTokenManager:
    # 설정과 테스트용 HTTP 클라이언트를 받아 토큰 관리 상태 초기화
    def __init__(
        self,
        app_settings: Settings = settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = app_settings
        self._http_client = http_client
        self._access_token: str | None = None
        self._refresh_at = 0.0
        self._lock = threading.Lock()

    # API 제공자를 확인하고 해당 제공자의 액세스 토큰 반환
    def get_access_token(self) -> str | None:
        provider = (self._settings.stock_api_provider or "").lower()

        if provider == "sample":
            return None
        if provider == "toss":
            return self._get_toss_access_token()

        raise ValueError(
            "Unsupported STOCK_API_PROVIDER: "
            f"{self._settings.stock_api_provider or '(not configured)'}"
        )

    # 캐시된 토큰을 제거하여 다음 요청에서 새 토큰 발급
    def invalidate_access_token(self) -> None:
        with self._lock:
            self._access_token = None
            self._refresh_at = 0.0

    # 유효한 캐시가 있으면 재사용하고 없으면 토스 토큰 발급
    def _get_toss_access_token(self) -> str:
        if self._has_valid_cached_token():
            return self._access_token or ""

        with self._lock:
            if self._has_valid_cached_token():
                return self._access_token or ""

            token, expires_in = self._issue_toss_access_token()
            refresh_buffer = min(TOKEN_REFRESH_BUFFER_SECONDS, expires_in // 2)

            self._access_token = token
            self._refresh_at = time.monotonic() + expires_in - refresh_buffer
            return token

    # 캐시된 토큰이 존재하고 예정된 재발급 시각 전인지 확인
    def _has_valid_cached_token(self) -> bool:
        return self._access_token is not None and time.monotonic() < self._refresh_at

    # 토스 OAuth API를 호출하고 액세스 토큰과 유효기간 반환
    def _issue_toss_access_token(self) -> tuple[str, int]:
        client_id = self._settings.stock_api_key
        client_secret = self._settings.stock_api_secret
        base_url = self._settings.stock_api_url

        if not client_id or not client_secret or not base_url:
            raise ValueError(
                "Toss API requires STOCK_API_KEY, STOCK_API_SECRET, and STOCK_API_URL."
            )

        request_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        token_url = f"{base_url.rstrip('/')}/oauth2/token"

        try:
            response = self._post_token_request(token_url, request_data)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            message = _extract_error_message(error.response)
            raise StockTokenError(
                f"Toss access token request failed ({error.response.status_code}): "
                f"{message}"
            ) from None
        except httpx.RequestError as error:
            raise StockTokenError(
                f"Toss access token request failed: {error.__class__.__name__}"
            ) from None

        try:
            response_data = response.json()
        except ValueError:
            raise StockTokenError(
                "Toss access token response is not valid JSON."
            ) from None

        access_token = response_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise StockTokenError(
                "Toss access token response does not contain access_token."
            )

        expires_in = _parse_expires_in(response_data.get("expires_in"))
        return access_token, expires_in

    # form-urlencoded 형식으로 토큰 발급 POST 요청 실행
    def _post_token_request(
        self,
        token_url: str,
        request_data: dict[str, str],
    ) -> httpx.Response:
        if self._http_client is not None:
            return self._http_client.post(token_url, data=request_data)

        return httpx.post(
            token_url,
            data=request_data,
            timeout=TOKEN_REQUEST_TIMEOUT_SECONDS,
        )


# API가 반환한 토큰 유효기간을 양의 정수로 변환
def _parse_expires_in(value: Any) -> int:
    try:
        expires_in = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_TTL_SECONDS

    if expires_in <= 0:
        return DEFAULT_TOKEN_TTL_SECONDS
    return expires_in


# 토스 API 오류 응답에서 사용자에게 표시할 메시지 추출
def _extract_error_message(response: httpx.Response) -> str:
    try:
        response_data = response.json()
    except ValueError:
        return "Unknown API error"

    error = response_data.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return "Unknown API error"


# 애플리케이션 전체에서 공유하는 토큰 매니저 인스턴스
stock_token_manager = StockTokenManager()


# 공유 토큰 매니저를 통해 현재 provider의 액세스 토큰 반환
def get_stock_access_token() -> str | None:
    return stock_token_manager.get_access_token()

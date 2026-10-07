from dataclasses import replace
from urllib.parse import parse_qs

import httpx
import pytest

from src.collectors.stock_token_manager import (
    StockTokenError,
    StockTokenManager,
)
from src.config.settings import settings


def build_settings(provider: str = "toss"):
    return replace(
        settings,
        stock_api_provider=provider,
        stock_api_key="test-client-id",
        stock_api_secret="test-client-secret",
        stock_api_url="https://openapi.tossinvest.com",
    )


def test_get_access_token_issues_and_caches_toss_token():
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "test-access-token",
                "token_type": "Bearer",
                "expires_in": 86400,
            },
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as client:
        manager = StockTokenManager(build_settings(), http_client=client)

        first_token = manager.get_access_token()
        second_token = manager.get_access_token()

    assert first_token == "test-access-token"
    assert second_token == "test-access-token"
    assert len(requests) == 1
    assert requests[0].url.path == "/oauth2/token"
    assert parse_qs(requests[0].content.decode()) == {
        "grant_type": ["client_credentials"],
        "client_id": ["test-client-id"],
        "client_secret": ["test-client-secret"],
    }


def test_get_access_token_returns_none_for_sample_provider():
    manager = StockTokenManager(build_settings(provider="sample"))

    assert manager.get_access_token() is None


def test_get_access_token_rejects_unsupported_provider():
    manager = StockTokenManager(build_settings(provider="unknown"))

    with pytest.raises(ValueError, match="Unsupported STOCK_API_PROVIDER"):
        manager.get_access_token()


def test_invalidate_access_token_requests_new_token():
    request_count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "access_token": f"test-access-token-{request_count}",
                "expires_in": 86400,
            },
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as client:
        manager = StockTokenManager(build_settings(), http_client=client)
        first_token = manager.get_access_token()
        manager.invalidate_access_token()
        second_token = manager.get_access_token()

    assert first_token == "test-access-token-1"
    assert second_token == "test-access-token-2"
    assert request_count == 2


def test_get_access_token_raises_for_toss_api_error():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid client credentials"}},
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as client:
        manager = StockTokenManager(build_settings(), http_client=client)

        with pytest.raises(StockTokenError, match="Invalid client credentials"):
            manager.get_access_token()

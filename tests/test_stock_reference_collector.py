from dataclasses import replace

import httpx
import pytest

from src.collectors.stock_reference_collector import refresh_stock_references
from src.config.settings import settings
from src.data_loaders.stock_reference_loader import load_stock_references


# 실제 자격증명 없이 토스 분기를 테스트할 설정 생성
def _test_settings():
    return replace(
        settings, stock_api_provider="toss", stock_api_key="test-id",
        stock_api_secret="test-secret", stock_api_url="https://example.test",
    )


# 두 시장 데이터의 통합 저장·로딩과 기존 파일 갱신 검증
def test_refresh_combines_markets_and_replaces_file(tmp_path):
    path = tmp_path / "reference" / "stock_name_code.json"
    requested_markets = []

    # 토큰과 각 시장의 종목 목록을 공식 응답 형식으로 반환
    def respond(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 86400})
        assert request.headers["Authorization"] == "Bearer test-token"
        market = request.url.params["market"]
        requested_markets.append(market)
        stock = (
            {"name": "삼성전자", "symbol": "005930"} if market == "KOSPI"
            else {"name": "JYP Ent.", "symbol": "035900"}
        )
        return httpx.Response(200, json={"result": [stock]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        records = refresh_stock_references(path, _test_settings(), http_client=client)
        path.write_text("old-data", encoding="utf-8")
        refresh_stock_references(path, _test_settings(), http_client=client)
    assert requested_markets == ["KOSPI", "KOSDAQ", "KOSPI", "KOSDAQ"]
    assert load_stock_references(path) == records
    assert records[0]["stock_code"] == "005930"
    assert {record["market"] for record in records} == {"KOSPI", "KOSDAQ"}


# 두 번째 시장의 실패 시 기존 파일을 덮어쓰지 않는지 검증
def test_failed_refresh_preserves_existing_file(tmp_path):
    path = tmp_path / "stock_name_code.json"
    path.write_text("existing-data", encoding="utf-8")

    # 첫 시장 성공 후 두 번째 시장에서 서버 오류 반환
    def respond(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 86400})
        if request.url.params["market"] == "KOSPI":
            return httpx.Response(200, json={"result": [{"name": "삼성전자", "symbol": "005930"}]})
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            refresh_stock_references(path, _test_settings(), http_client=client)
    assert path.read_text(encoding="utf-8") == "existing-data"


# sample provider의 중복 제거 및 미지원 provider 거부 검증
def test_provider_branches(tmp_path):
    path = tmp_path / "stock_name_code.json"
    records = refresh_stock_references(path, replace(settings, stock_api_provider="sample"))
    assert len(records) == 4
    assert all(record["market"] == "SAMPLE" for record in records)
    with pytest.raises(ValueError, match="Unsupported STOCK_API_PROVIDER"):
        refresh_stock_references(path, replace(settings, stock_api_provider="unknown"))

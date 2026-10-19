from dataclasses import replace

import httpx
import pytest

from src.collectors.stock_reference_collector import refresh_stock_references
from src.config.settings import settings
from src.data_loaders.stock_reference_loader import load_stock_references


# 실제 대기 대신 대기 시간을 기록하여 호출 간격과 재시도 검증
@pytest.fixture(autouse=True)
def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr("src.collectors.stock_reference_collector.time.sleep", calls.append)
    return calls


# 실제 자격증명 없이 토스 분기를 테스트할 설정 생성
def _test_settings():
    return replace(
        settings, stock_api_provider="toss", stock_api_key="test-id",
        stock_api_secret="test-secret", stock_api_url="https://example.test",
    )


# 두 시장 데이터의 통합 저장·로딩과 기존 파일 갱신 검증
def test_refresh_combines_markets_and_replaces_file(tmp_path, sleep_calls):
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
    assert sleep_calls == [1.1, 1.1]


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


# 일시적 429 발생 시 해당 시장만 재시도하고 서버 권장 대기 시간 준수
@pytest.mark.parametrize("retry_after,minimum_delay", [("2", 2), ("invalid", 1.1)])
def test_rate_limit_retry_succeeds(tmp_path, sleep_calls, retry_after, minimum_delay):
    markets = []

    # 첫 KOSDAQ 요청은 제한하고 다음 요청은 정상 응답
    def respond(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 86400})
        market = request.url.params["market"]
        markets.append(market)
        if market == "KOSDAQ" and markets.count(market) == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after})
        stock = (
            {"name": "삼성전자", "symbol": "005930"} if market == "KOSPI"
            else {"name": "JYP Ent.", "symbol": "035900"}
        )
        return httpx.Response(200, json={"result": [stock]})

    path = tmp_path / "stocks.json"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        records = refresh_stock_references(path, _test_settings(), http_client=client)
    assert markets == ["KOSPI", "KOSDAQ", "KOSDAQ"]
    assert sleep_calls[0] == 1.1
    assert minimum_delay <= sleep_calls[1] <= minimum_delay + 0.3
    assert load_stock_references(path) == records


# 지속적인 429는 최초 요청과 재시도 3회 이후 중단하고 기존 파일 보존
def test_rate_limit_retry_is_bounded(tmp_path, sleep_calls):
    requests = []

    # 토큰은 정상 발급하고 종목 목록은 계속 요청 제한 응답
    def respond(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json={"access_token": "test-token", "expires_in": 86400})
        requests.append(request)
        return httpx.Response(429)

    path = tmp_path / "stocks.json"
    path.write_text("existing-data", encoding="utf-8")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            refresh_stock_references(path, _test_settings(), http_client=client)
    assert len(requests) == 4
    assert len(sleep_calls) == 3
    assert path.read_text(encoding="utf-8") == "existing-data"

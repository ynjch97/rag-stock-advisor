from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

from openai import OpenAIError
import pytest

from src.config.settings import settings
from src.domain.advice_models import StockDiscoveryCriteria, StockDiscoveryEvidence
from src.parsers.llm_stock_discovery_analyzer import analyze_discovery_candidates


# 실제 네트워크 없이 강제 Tool Calling 응답을 제공
@pytest.fixture
def client():
    connection = Mock()
    connection.with_options.return_value = connection
    connection.responses.create.return_value = SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="function_call", name="submit_discovery_candidates", arguments='{"candidates": []}')])
    return connection


# 테스트에 필요한 최소 원문 근거
@pytest.fixture
def articles():
    return [StockDiscoveryEvidence(title="소재 사업", url="https://example.com/news/1", content="본문" * 1500)]


# 출처 번호·본문 길이·재시도 금지·reasoning 설정과 strict 스키마 검증
def test_request_contract(client, articles):
    config = replace(settings, openai_model="gpt-5-mini", openai_reasoning="minimal")
    result = analyze_discovery_candidates(StockDiscoveryCriteria("반도체 소재", 3), articles, 10,
                                          client=client, app_settings=config)
    assert result.candidates == []
    client.responses.create.assert_called_once()
    options = client.responses.create.call_args.kwargs
    data = json.loads(options["input"])
    assert data["candidate_limit"] == 10
    assert data["criteria"]["requested_count"] == 3
    assert data["sources"][0]["source_index"] == 0
    assert len(data["sources"][0]["content"]) == 2000
    assert options["reasoning"] == {"effort": "minimal"}
    assert options["parallel_tool_calls"] is False
    assert options["store"] is False
    assert client.with_options.call_args.kwargs["max_retries"] == 0
    schema = options["tools"][0]["parameters"]
    for model in [schema, *schema["$defs"].values()]:
        assert model["additionalProperties"] is False
        assert set(model["required"]) == set(model["properties"])


# 본문 근거가 없으면 LLM을 호출하지 않음
def test_empty_sources_skip_llm(client):
    assert not analyze_discovery_candidates(StockDiscoveryCriteria("소재"), [], 10, client=client).candidates
    client.responses.create.assert_not_called()


# 불완전 응답·다른 도구·스키마 위반은 임의 보완하지 않음
@pytest.mark.parametrize("status,name,arguments", [
    ("incomplete", "submit_discovery_candidates", '{"candidates": []}'),
    ("completed", "other_tool", '{"candidates": []}'),
    ("completed", "submit_discovery_candidates", '{"candidates": [], "extra": 1}'),
    ("completed", "submit_discovery_candidates", 'not-json'),
])
def test_invalid_response(client, articles, status, name, arguments):
    client.responses.create.return_value = SimpleNamespace(status=status, output=[
        SimpleNamespace(type="function_call", name=name, arguments=arguments)])
    with pytest.raises(ValueError):
        analyze_discovery_candidates(StockDiscoveryCriteria("소재"), articles, 10, client=client)


# 서버 후보 수 상한을 초과하는 응답은 거부
def test_excess_candidates(client, articles):
    candidate = {"stock_name": "기업A", "selection_reason": "사업 연관",
                 "evidence": [{"source_index": 0, "excerpt": "사업", "relation": "direct"}]}
    client.responses.create.return_value.output[0].arguments = json.dumps({"candidates": [candidate] * 2})
    with pytest.raises(ValueError, match="candidate limit"):
        analyze_discovery_candidates(StockDiscoveryCriteria("소재"), articles, 1, client=client)


# 자격정보 부족·잘못된 상한은 외부 요청 전에 거부
@pytest.mark.parametrize("limit", [0, 11, True])
def test_invalid_limit(client, articles, limit):
    with pytest.raises(ValueError):
        analyze_discovery_candidates(StockDiscoveryCriteria("소재"), articles, limit, client=client)
    client.responses.create.assert_not_called()


# 에러 메시지에 SDK의 원문 응답이나 인증정보를 노출하지 않음
def test_sanitized_api_error(client, articles):
    client.responses.create.side_effect = OpenAIError("private response")
    with pytest.raises(ValueError, match="OpenAIError") as error:
        analyze_discovery_candidates(StockDiscoveryCriteria("소재"), articles, 10, client=client)
    assert "private response" not in str(error.value)


# API 키가 없으면 클라이언트를 만들지 않음
def test_missing_key(articles):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        analyze_discovery_candidates(StockDiscoveryCriteria("소재"), articles, 10,
                                    app_settings=replace(settings, openai_api_key=None))

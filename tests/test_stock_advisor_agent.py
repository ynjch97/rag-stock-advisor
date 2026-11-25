from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.agents.stock_advisor_agent import decide_next_action
from src.config.settings import settings
from src.domain.advice_models import (ParsedQuestion, ParsedStock, StockDiscoveryCriteria,
                                      StockDiscoveryCandidate, StockDiscoveryEvidence)


# 실제 API 키·통신 없이 strict Tool Calling 응답을 구성
@pytest.fixture
def environment():
    stock = ParsedStock("하이닉스", "SK하이닉스", "000660")
    parsed = ParsedQuestion("하이닉스 전망", (stock,), "general_analysis")
    client = Mock()
    client.with_options.return_value = client
    output = {"action": "collect_news", "reason": "관련 뉴스 없음", "stock_code": "000660",
              "search_query": None, "news_start": 1}
    config = replace(settings, openai_api_key=None, openai_model="gpt-5-mini", openai_reasoning="low")
    allowed = [{"action": "collect_news", "stock_code": "000660", "news_start": 1}, {"action": "respond"}]
    return parsed, client, output, config, allowed


# 지정 응답을 SDK 대역에 설정하여 결정 함수 호출
def _decide(environment, **overrides):
    parsed, client, output, config, allowed = environment
    client.responses.create.return_value = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="function_call", name="submit_agent_decision", arguments=json.dumps({**output, **overrides}),
    )])
    return decide_next_action(parsed, [{"news": []}], [], allowed, app_settings=config, client=client)


# 검증된 종목·단일 Tool Calling·reasoning·SDK 재시도 비활성화 확인
def test_agent_collect_news_decision(environment):
    decision = _decide(environment)
    assert decision.stock == environment[0].stocks[0]
    assert decision.news_start == 1
    options = environment[1].responses.create.call_args.kwargs
    assert options["parallel_tool_calls"] is False and options["store"] is False
    assert options["tools"][0]["strict"] is True
    assert options["reasoning"] == {"effort": "low"}
    assert environment[1].with_options.call_args.kwargs["max_retries"] == 0


# 임의 종목·허용되지 않은 페이지·다른 도구 인자·잘못된 구조 거부
@pytest.mark.parametrize("overrides", [
    {"stock_code": "005930"}, {"news_start": 31}, {"news_start": 61}, {"reason": ""},
    {"action": "unknown"}, {"search_query": "임의 표현"},
])
def test_invalid_agent_decision(environment, overrides):
    with pytest.raises(ValueError):
        _decide(environment, **overrides)


# 응답·재검색 결정의 조건부 인자 확인
def test_agent_terminal_and_rewritten_query(environment):
    assert _decide(environment, action="respond", stock_code=None, news_start=None).stock is None
    environment[-1].append({"action": "search_news", "stock_code": "000660"})
    result = _decide(environment, action="search_news", news_start=None, search_query="SK하이닉스 실적")
    assert result.search_query == "SK하이닉스 실적"
    with pytest.raises(ValueError):
        _decide(environment, action="search_news", news_start=None, search_query=" ")


# 종료 결정의 불필요한 인자는 제거하되 허용되지 않은 종료 작업은 거부
@pytest.mark.parametrize("action", ["respond", "stop"])
@pytest.mark.parametrize("arguments", [
    {"stock_code": "005930", "search_query": None, "news_start": None},
    {"stock_code": None, "search_query": "이전 검색어", "news_start": None},
    {"stock_code": None, "search_query": None, "news_start": 31},
    {"stock_code": "000660", "search_query": "", "news_start": 1},
])
def test_terminal_decisions_ignore_unused_arguments(environment, action, arguments):
    environment[-1].append({"action": action})
    decision = _decide(environment, action=action, **arguments)
    assert decision.action == action
    assert decision.reason == environment[2]["reason"]
    assert decision.stock is None and decision.search_query is None and decision.news_start is None


@pytest.mark.parametrize("action", ["respond", "stop"])
def test_terminal_decisions_still_require_allowed_action(environment, action):
    environment[-1][:] = [{"action": "collect_news", "stock_code": "000660", "news_start": 1}]
    with pytest.raises(ValueError, match="outside the allowed budget"):
        _decide(environment, action=action)


# 외부 API 오류는 본문·인증정보를 노출하지 않고 종료 사유로 변환
def test_agent_api_error_is_sanitized(environment):
    client = environment[1]
    client.responses.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    with pytest.raises(ValueError, match="APIConnectionError"):
        _decide(environment)


# 미완료·잘못된 도구·다중 호출 응답 거부
@pytest.mark.parametrize("status, name, count", [("incomplete", "submit_agent_decision", 1),
                                                ("completed", "other", 1), ("completed", "submit_agent_decision", 2)])
def test_invalid_tool_response(environment, status, name, count):
    parsed, client, output, config, allowed = environment
    client.responses.create.return_value = SimpleNamespace(status=status, output=[SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(output),
    )] * count)
    with pytest.raises(ValueError):
        decide_next_action(parsed, [{"news": []}], [], allowed, app_settings=config, client=client)


# 발굴 단계에서는 종목코드 없이 허용된 페이지의 작업을 선택
@pytest.mark.parametrize("page", [1, 31])
def test_discovery_decision(environment, page):
    _, client, _, config, _ = environment
    parsed = ParsedQuestion("반도체 관련주", (), "stock_discovery", discovery=StockDiscoveryCriteria("반도체"))
    output = {"action": "discover_stocks", "reason": "후보 부족", "stock_code": None,
              "search_query": None, "news_start": page}
    client.responses.create.return_value = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="function_call", name="submit_agent_decision", arguments=json.dumps(output))])
    decision = decide_next_action(parsed, [], [], [{"action": "discover_stocks", "news_start": page}],
                                 app_settings=config, client=client)
    assert decision.stock is None and decision.news_start == page
    schema = client.responses.create.call_args.kwargs["tools"][0]["parameters"]
    assert schema["properties"]["action"]["enum"] == ["discover_stocks"]
    for arguments in ({"stock_code": "000660"}, {"search_query": "임의 테마"}, {"news_start": None},
                      {"news_start": 31 if page == 1 else 1}):
        client.responses.create.return_value.output[0].arguments = json.dumps({**output, **arguments})
        with pytest.raises(ValueError):
            decide_next_action(parsed, [], [], [{"action": "discover_stocks", "news_start": page}],
                               app_settings=config, client=client)


# 검증된 후보만 분석 대상으로 전달하며 질문 stocks와 구분하고 원문 전체 재전송 방지
def test_discovered_candidate_provenance(environment):
    _, client, _, config, _ = environment
    candidate = StockDiscoveryCandidate("SK하이닉스", "원문 근거", stock_code="000660", business_verified=True,
        evidence=(StockDiscoveryEvidence("사업 기사", "https://example.test/1", "보존할 본문", excerpt="사업 인용"),))
    parsed = ParsedQuestion("관련주 추천", (), "stock_discovery", discovery=StockDiscoveryCriteria("반도체"),
                            discovered_candidates=(candidate,))
    output = {"action": "collect_stock", "reason": "주가 부족", "stock_code": "000660",
              "search_query": None, "news_start": None}
    client.responses.create.return_value = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="function_call", name="submit_agent_decision", arguments=json.dumps(output))])
    context = {"news": [], "discovery_candidate": candidate.to_dict()}
    decision = decide_next_action(parsed, [context], [], [{"action": "collect_stock", "stock_code": "000660"}],
                                 app_settings=config, client=client)
    assert decision.stock.role == "discovered"
    data = json.loads(client.responses.create.call_args.kwargs["input"])
    assert data["question"]["stocks"] == []
    assert data["analysis_stocks"][0]["stock_code"] == "000660"
    assert "content" not in data["contexts"][0]["discovery_candidate"]["evidence"][0]
    assert context["discovery_candidate"]["evidence"][0]["content"] == "보존할 본문"

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.agents.stock_advisor_agent import decide_next_action
from src.config.settings import settings
from src.domain.advice_models import (ParsedQuestion, ParsedStock, StockDiscoveryCandidate,
                                      StockDiscoveryCriteria, StockDiscoveryEvidence)


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
    {"action": "respond", "news_start": None},
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


# 종목 미지정 발굴 질문과 코드 없는 발굴 작업을 허용
@pytest.fixture
def discovery_environment(environment):
    values = list(environment)
    values[0] = ParsedQuestion("반도체 소재 관련주 추천", (), "stock_discovery",
                               discovery=StockDiscoveryCriteria("반도체 소재", 3))
    values[-1] = [{"action": "discover_stocks", "news_start": 1}, {"action": "stop"}]
    return values


# 발굴은 LLM이 생성한 종목코드 없이 질문의 발굴 조건만으로 선택
def test_agent_selects_discovery(discovery_environment):
    decision = _decide(discovery_environment, action="discover_stocks", stock_code=None, news_start=1)
    assert decision.action == "discover_stocks"
    assert decision.stock is None
    request = discovery_environment[1].responses.create.call_args.kwargs
    assert json.loads(request["input"])["question"]["discovery"]["theme"] == "반도체 소재"


# 발굴 시 임의 종목·검색어·미허용 페이지나 페이지 누락을 차단
@pytest.mark.parametrize("overrides", [
    {"stock_code": "000660"}, {"search_query": "다른 테마"}, {"news_start": 31}, {"news_start": None},
])
def test_discovery_rejects_tool_arguments(discovery_environment, overrides):
    arguments = {"action": "discover_stocks", "stock_code": None, "news_start": 1, **overrides}
    with pytest.raises(ValueError):
        _decide(discovery_environment, **arguments)


# 일반 종목 질문이나 누락된 발굴 조건은 발굴 작업으로 처리하지 않음
def test_discovery_requires_criteria(environment, discovery_environment):
    environment[-1].append({"action": "discover_stocks", "news_start": 1})
    with pytest.raises(ValueError, match="Discovery requires"):
        _decide(environment, action="discover_stocks", stock_code=None, news_start=1)
    discovery_environment[0] = replace(discovery_environment[0], discovery=None)
    with pytest.raises(ValueError, match="Discovery requires"):
        _decide(discovery_environment, action="discover_stocks", stock_code=None, news_start=1)


# 발굴 후보를 질문 속 종목에 주입하지 않고 검증된 분석 대상으로만 허용
def test_agent_targets_verified_discovery_candidate(discovery_environment):
    candidate = StockDiscoveryCandidate("SK하이닉스", "기사 근거", stock_code="000660", business_verified=True,
        evidence=(StockDiscoveryEvidence("기사", "https://example.test/1", "전체 본문", excerpt="실제 인용"),))
    discovery_environment[0] = replace(discovery_environment[0], discovered_candidates=(candidate,))
    discovery_environment[-1].append({"action": "collect_stock", "stock_code": "000660"})
    result = _decide(discovery_environment, action="collect_stock", news_start=None)
    assert result.stock.role == "discovered"
    request = json.loads(discovery_environment[1].responses.create.call_args.kwargs["input"])
    assert request["question"]["stocks"] == []
    assert request["analysis_stocks"][0]["stock_code"] == "000660"
    with pytest.raises(ValueError, match="Discovery requires"):
        _decide(discovery_environment, action="discover_stocks", stock_code=None, news_start=None)


# 발굴 본문 전체를 반복 전송하지 않고 인용·출처만 전달하며 원본 컨텍스트는 보존
def test_agent_discovery_evidence_is_bounded(discovery_environment):
    candidate = StockDiscoveryCandidate("SK하이닉스", "기사 근거", stock_code="000660", business_verified=True,
        evidence=(StockDiscoveryEvidence("기사", "https://example.test/1", "보존할 전체 본문", excerpt="실제 인용"),))
    discovery_environment[0] = replace(discovery_environment[0], discovered_candidates=(candidate,))
    _decide(discovery_environment, action="stop", stock_code=None, news_start=None)
    parsed, client, _, config, _ = discovery_environment
    context = {"news": [], "discovery_candidate": candidate.to_dict()}
    decide_next_action(parsed, [context], [], [{"action": "stop"}], app_settings=config, client=client)
    request = json.loads(client.responses.create.call_args.kwargs["input"])
    evidence = request["contexts"][0]["discovery_candidate"]["evidence"][0]
    assert evidence["excerpt"] == "실제 인용"
    assert "content" not in evidence
    assert context["discovery_candidate"]["evidence"][0]["content"] == "보존할 전체 본문"

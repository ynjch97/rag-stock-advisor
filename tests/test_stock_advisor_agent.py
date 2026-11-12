from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.agents.stock_advisor_agent import decide_next_action
from src.config.settings import settings
from src.domain.advice_models import ParsedQuestion, ParsedStock


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

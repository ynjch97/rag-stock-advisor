from dataclasses import replace
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.config.settings import settings
from src.parsers.llm_question_analyzer import TOOL_NAME, analyze_stock_question


# 실제 응답 스키마를 따르는 모의 도구 인자
@pytest.fixture
def analysis_payload():
    return {
        "stocks": [{"mention": "삼전", "stock_name": "삼성전자", "role": "subject"}],
        "intent": "buy_opinion",
        "period": {"kind": "default", "recent_days": None, "start_date": None, "end_date": None},
        "needs_clarification": False, "clarification_question": None,
    }


# 주입한 OpenAI 클라이언트가 구조화된 함수 호출을 반환하도록 설정
def _client(payload):
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        status="completed", output=[SimpleNamespace(
            type="function_call", name=TOOL_NAME, arguments=json.dumps(payload),
        )],
    )
    return client


# 강제 Tool Calling·strict 스키마·한국 기준일·설정 모델 전달 검증
def test_analyzer_uses_strict_function_calling(analysis_payload):
    client = _client(analysis_payload)
    config = replace(settings, openai_model="test-model", openai_api_key=None)
    result = analyze_stock_question("삼전 사도 돼?", date(2026, 10, 26), config, client)
    assert result.stocks[0].stock_name == "삼성전자"
    arguments = client.responses.create.call_args.kwargs
    assert arguments["model"] == "test-model"
    assert arguments["tool_choice"] == {"type": "function", "name": TOOL_NAME}
    assert arguments["parallel_tool_calls"] is False
    assert arguments["store"] is False
    assert "2026-10-26" in arguments["instructions"]
    tool = arguments["tools"][0]
    assert tool["strict"] is True
    schema = tool["parameters"]
    assert set(schema["required"]) == set(schema["properties"])
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])
    client.close.assert_not_called()


# 키 누락 시 네트워크 호출 전에 설정 안내
def test_missing_key_is_actionable():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        analyze_stock_question("삼성전자", date(2026, 10, 26), replace(settings, openai_api_key=None))


# 불완전 응답·도구 없음·잘못된 도구·중복 호출·유효하지 않은 JSON 거부
@pytest.mark.parametrize("failure", ["incomplete", "missing", "wrong_tool", "duplicate", "json", "schema"])
def test_invalid_tool_response_is_rejected(analysis_payload, failure):
    client = _client(analysis_payload)
    response = client.responses.create.return_value
    if failure == "incomplete":
        response.status = "incomplete"
    elif failure == "missing":
        response.output = []
    elif failure == "wrong_tool":
        response.output[0].name = "execute_arbitrary_code"
    elif failure == "duplicate":
        response.output *= 2
    elif failure == "json":
        response.output[0].arguments = "not-json"
    else:
        analysis_payload["unexpected"] = "field"
        response.output[0].arguments = json.dumps(analysis_payload)
    with pytest.raises(ValueError):
        analyze_stock_question("삼전", date(2026, 10, 26), client=client)


# OpenAI 오류를 규칙 기반 해석으로 대체하지 않고 예외로 전달
def test_api_failure_has_no_rule_fallback():
    client = Mock()
    client.responses.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    with pytest.raises(ValueError, match="OpenAI 질문 분석 요청 실패"):
        analyze_stock_question("삼전", date(2026, 10, 26), client=client)

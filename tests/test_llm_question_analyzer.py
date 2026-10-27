from dataclasses import replace
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.config.settings import load_settings, settings
from src.parsers import llm_question_analyzer as analyzer
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


# 환경변수 추론 강도를 지원 모델 요청에 그대로 전달
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_reasoning_effort_is_forwarded(analysis_payload, effort):
    client = _client(analysis_payload)
    config = replace(settings, openai_model="gpt-5-mini", openai_reasoning=effort)
    result = analyze_stock_question("삼전", date(2026, 10, 26), config, client)
    assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": effort}
    assert result.openai_call_attempts == 1
    assert "openai_call_attempts" not in analyzer.QuestionAnalysis.model_json_schema()["properties"]


# 추론 미지원 모델의 기존 요청 형식 유지
def test_non_reasoning_model_omits_reasoning(analysis_payload):
    client = _client(analysis_payload)
    config = replace(settings, openai_model="gpt-4o")
    analyze_stock_question("삼전", date(2026, 10, 26), config, client)
    assert "reasoning" not in client.responses.create.call_args.kwargs


# .env 값 로딩과 시스템 환경변수 우선순위 검증
def test_reasoning_setting_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_REASONING", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_REASONING=low\n", encoding="utf-8")
    assert load_settings(env_path).openai_reasoning == "low"
    monkeypatch.setenv("OPENAI_REASONING", "high")
    assert load_settings(env_path).openai_reasoning == "high"
    monkeypatch.delenv("OPENAI_REASONING")
    env_path.write_text("", encoding="utf-8")
    assert load_settings(env_path).openai_reasoning == "medium"


# 실제 SDK의 재시도 요청을 집계하고 반복 질문에서 연결 재사용·횟수 초기화 검증
def test_client_reuse_and_retry_attempt_count(analysis_payload, monkeypatch):
    requests = []

    # 첫 요청만 429를 반환하여 SDK 자동 재시도 유도
    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, json={
                "error": {"message": "retry", "type": "rate_limit_error"},
            })
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-5-mini", "status": "completed",
            "output": [{"id": "fc_test", "type": "function_call", "call_id": "call_test",
                        "name": TOOL_NAME, "arguments": json.dumps(analysis_payload)}],
        })

    # 외부 네트워크 대신 모의 전송 계층을 사용하는 실제 HTTP 클라이언트 생성
    def make_http_client(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(respond), **kwargs)

    factory = Mock(side_effect=make_http_client)
    monkeypatch.setattr(analyzer, "DefaultHttpxClient", factory)
    analyzer._get_openai_client.cache_clear()
    config = replace(settings, openai_api_key="test-retry-key", openai_model="gpt-5-mini")
    try:
        first = analyze_stock_question("삼전", date(2026, 10, 26), config)
        connection = analyzer._get_openai_client(config.openai_api_key)
        second = analyze_stock_question("삼전", date(2026, 10, 26), config)
        assert first.openai_call_attempts == 2
        assert second.openai_call_attempts == 1
        assert len(requests) == 3
        assert analyzer._get_openai_client(config.openai_api_key) is connection
        assert not connection.is_closed()
        factory.assert_called_once()
    finally:
        analyzer._get_openai_client(config.openai_api_key).close()
        analyzer._get_openai_client.cache_clear()

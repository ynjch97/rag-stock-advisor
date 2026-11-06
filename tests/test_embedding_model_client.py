from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from src.config.settings import settings
from src.embeddings import embedding_model_client as model_client


# 서버 응답 순서가 달라도 입력 index에 맞춰 벡터를 반환하며 주입 클라이언트는 닫지 않음
def test_request_restores_input_order():
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(data=[
        SimpleNamespace(index=1, embedding=[0.0, 1.0]), SimpleNamespace(index=0, embedding=[1.0, 0.0]),
    ])
    config = replace(settings, openai_api_key=None, openai_embedding_model="test-embedding")
    vectors = model_client.request_embeddings(["first", "second"], app_settings=config, client=client)
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    client.embeddings.create.assert_called_once_with(
        model="test-embedding", input=["first", "second"], encoding_format="float",
    )
    client.close.assert_not_called()


# 누락·중복·범위 밖 index는 문서와 벡터의 잘못된 연결을 막기 위해 거부
@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2]])
def test_invalid_response_indices(indices):
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(data=[
        SimpleNamespace(index=index, embedding=[1.0]) for index in indices
    ])
    with pytest.raises(ValueError, match="indices"):
        model_client.request_embeddings(["a", "b"], client=client)


# 키가 없으면 SDK 생성이나 실제 요청을 하지 않음
def test_missing_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model_client.request_embeddings(["news"], app_settings=replace(settings, openai_api_key=None))


# SDK 오류 내용의 인증정보·응답 본문은 사용자 예외에 노출하지 않음
def test_sdk_failure_is_sanitized():
    client = Mock()
    client.embeddings.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    with pytest.raises(ValueError, match="APIConnectionError"):
        model_client.request_embeddings(["news"], client=client)


# 같은 키의 반복 요청에서 하나의 클라이언트 재사용
def test_client_is_cached(monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(model_client, "OpenAI", factory)
    model_client._get_embedding_client.cache_clear()
    try:
        assert model_client._get_embedding_client("dummy-key") is client
        assert model_client._get_embedding_client("dummy-key") is client
        factory.assert_called_once()
    finally:
        model_client._get_embedding_client.cache_clear()

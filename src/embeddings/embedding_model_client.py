import atexit
from functools import lru_cache

from openai import OpenAI, OpenAIError

from src.config.settings import Settings, settings
from src.services.execution_budget import remaining_timeout


# 같은 API 키의 클라이언트를 재사용하고 프로세스 종료 시 연결 정리
@lru_cache(maxsize=None)
def _get_embedding_client(api_key: str) -> OpenAI:
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
    atexit.register(client.close)
    return client


# 텍스트 묶음을 모델에 전달하고 응답 index에 맞춰 입력 순서대로 벡터 반환
def request_embeddings(
    texts: list[str], *, app_settings: Settings = settings, client: OpenAI | None = None,
    deadline: float | None = None,
) -> list[list[float]]:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be nonempty texts.")
    if client is None and not app_settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings.")
    connection = client if client is not None else _get_embedding_client(app_settings.openai_api_key)
    if deadline is not None:
        connection = connection.with_options(
            timeout=remaining_timeout(deadline, app_settings.agent_request_timeout_seconds), max_retries=0,
        )
    try:
        response = connection.embeddings.create(
            model=app_settings.openai_embedding_model, input=texts, encoding_format="float",
        )
    except OpenAIError as error:
        raise ValueError(f"OpenAI embedding request failed: {type(error).__name__}") from None
    indices = [item.index for item in response.data]
    if sorted(indices) != list(range(len(texts))):
        raise ValueError("Embedding response indices do not match the inputs.")
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

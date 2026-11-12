import numpy as np
from openai import OpenAI

from src.config.settings import Settings, settings
from src.domain.news_models import NewsChunk
from src.embeddings.embedding_model_client import request_embeddings


DEFAULT_EMBEDDING_BATCH_SIZE = 32


# FAISS와 BM25가 동일하게 사용할 종목명·기사 제목·본문 텍스트 구성
def build_news_index_texts(chunks: list[NewsChunk]) -> list[str]:
    return [f"종목: {chunk.stock_name}\n제목: {chunk.title}\n{chunk.content}" for chunk in chunks]


# 텍스트를 묶음 요청하고 개수·차원·유한값을 검증한 float32 행렬 반환
def embed_texts(
    texts: list[str], *, app_settings: Settings = settings, client: OpenAI | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    deadline: float | None = None,
) -> np.ndarray:
    if type(batch_size) is not int or not 1 <= batch_size <= 2048:
        raise ValueError("Embedding batch_size must be between 1 and 2048.")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be nonempty texts.")
    batches = []
    dimension = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            options = {"deadline": deadline} if deadline is not None else {}
            vectors = np.asarray(request_embeddings(batch, app_settings=app_settings, client=client, **options), dtype=np.float32)
        except (TypeError, OverflowError):
            raise ValueError("Embedding vectors must be numeric arrays.") from None
        if (vectors.ndim != 2 or vectors.shape[0] != len(batch) or vectors.shape[1] == 0
                or not np.isfinite(vectors).all()):
            raise ValueError("Invalid embedding count, dimension, or numeric value.")
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError("Embedding dimensions differ between batches.")
        dimension = vectors.shape[1]
        batches.append(vectors)
    return np.ascontiguousarray(np.concatenate(batches))

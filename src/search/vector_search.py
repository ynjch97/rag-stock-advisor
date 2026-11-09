from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from src.config.paths import PROJECT_ROOT
from src.config.settings import Settings, settings
from src.domain.news_models import NewsChunk
from src.embeddings.embedding_service import embed_texts
from src.indexing.faiss_index import load_faiss_index


# 기존 FAISS와 청크 매핑을 로딩하고 저장 파일의 일치 여부 검증
def _load_vector_index(directory: Path) -> tuple[faiss.Index, list[NewsChunk], dict]:
    with (directory / "stock_advisor_metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or not isinstance(metadata.get("configuration"), dict)
            or metadata["configuration"].get("metric") != "cosine"
            or not isinstance(metadata.get("chunks"), list)):
        raise ValueError("Invalid FAISS news metadata; rebuild the news index.")
    path = directory / "stock_advisor.index"
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata.get("faiss_sha256"):
        raise ValueError("FAISS index and metadata do not match; rebuild the news index.")
    index = load_faiss_index(path)
    try:
        chunks = [NewsChunk(**record) for record in metadata["chunks"]]
        ids = [chunk.chunk_id for chunk in chunks]
        for chunk in chunks:
            if (any(not isinstance(value, str) or not value.strip() for value in (
                chunk.chunk_id, chunk.news_id, chunk.stock_name, chunk.stock_code, chunk.url,
                chunk.title, chunk.content, chunk.publisher, chunk.published_date,
            )) or chunk.important not in ("O", "X")
                    or type(chunk.chunk_index) is not int or chunk.chunk_index < 1
                    or type(chunk.chunk_count) is not int or chunk.chunk_count < chunk.chunk_index
                    or type(chunk.start_index) is not int or chunk.start_index < 0):
                raise ValueError("Invalid saved news chunk fields.")
            date.fromisoformat(chunk.published_date)
    except (TypeError, ValueError):
        raise ValueError("Invalid saved news chunks; rebuild the news index.") from None
    if (index.ntotal != len(chunks) or len(set(ids)) != len(ids)
            or metadata.get("chunk_count") != len(chunks) or metadata.get("dimension") != index.d):
        raise ValueError("FAISS vectors and news chunks do not match; rebuild the news index.")
    return index, chunks, metadata


# 질문을 임베딩하여 기존 인덱스에서 검색하고 코사인 점수와 청크 정보를 반환
def search_news_vectors(
    question: str, *, top_k: int = 5, stock_code: str | None = None,
    start_date: str | None = None, end_date: str | None = None,
    min_score: float | None = None, faiss_dir: Path | None = None,
    app_settings: Settings = settings, client: OpenAI | None = None,
) -> list[dict]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if min_score is not None and (type(min_score) not in (int, float)
                                 or not math.isfinite(min_score) or not -1 <= min_score <= 1):
        raise ValueError("min_score must be between -1 and 1.")
    if stock_code is not None and (not isinstance(stock_code, str) or not stock_code.strip()):
        raise ValueError("stock_code must be a nonempty string.")
    start = date.fromisoformat(start_date) if start_date is not None else None
    end = date.fromisoformat(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date.")
    if app_settings.vector_db_provider.lower() != "faiss":
        raise ValueError("Vector search requires VECTOR_DB_PROVIDER=faiss.")
    directory = faiss_dir if faiss_dir is not None else PROJECT_ROOT / app_settings.vector_db_path
    index, chunks, metadata = _load_vector_index(directory)
    if (metadata.get("embedding_model") != app_settings.openai_embedding_model
            or metadata["configuration"].get("embedding_model") != app_settings.openai_embedding_model):
        raise ValueError("Embedding model differs from the saved index; use the same model or rebuild.")
    candidates = {
        position for position, chunk in enumerate(chunks)
        if (stock_code is None or chunk.stock_code == stock_code)
        and (start is None or date.fromisoformat(chunk.published_date) >= start)
        and (end is None or date.fromisoformat(chunk.published_date) <= end)
    }
    if not candidates:
        return []
    query = embed_texts([question.strip()], app_settings=app_settings, client=client)
    if query.shape[1] != index.d:
        raise ValueError("Question embedding dimension differs from the saved index.")
    norm = np.linalg.norm(query.astype(np.float64))
    if norm == 0:
        raise ValueError("Question embedding must not be a zero vector.")
    query /= norm
    faiss.normalize_L2(query)
    # 필터가 있으면 전체 순위에서 조건에 맞는 상위 k개를 선택하여 결과 누락 방지
    count = min(top_k, index.ntotal) if len(candidates) == len(chunks) else index.ntotal
    scores, positions = index.search(query, count)
    results = []
    for score, position in zip(scores[0], positions[0]):
        if int(position) not in candidates:
            continue
        score = float(np.clip(score, -1.0, 1.0))
        if min_score is not None and score < min_score:
            continue
        results.append({**asdict(chunks[int(position)]), "score": score})
        if len(results) == top_k:
            break
    return results

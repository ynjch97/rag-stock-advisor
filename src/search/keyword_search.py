from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from src.config.paths import BM25_TOKENS_PATH, PROJECT_ROOT
from src.config.settings import Settings, settings
from src.domain.news_models import NewsChunk
from src.indexing.bm25_index import load_bm25_index, tokenize_text


# FAISS 바이너리 없이 BM25 토큰 파일과 청크 메타데이터의 체크섬·순서 검증
def _load_keyword_index(
    bm25_path: Path, metadata_path: Path,
) -> tuple[BM25Okapi, list[NewsChunk], dict]:
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or not isinstance(metadata.get("chunks"), list)
            or not isinstance(metadata.get("fingerprint"), str) or not metadata["fingerprint"]):
        raise ValueError("Invalid BM25 news metadata; rebuild the news index.")
    digest = hashlib.sha256()
    with bm25_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata.get("bm25_sha256"):
        raise ValueError("BM25 tokens and metadata do not match; rebuild the news index.")
    bm25, corpus = load_bm25_index(bm25_path)
    try:
        chunks = [NewsChunk(**record) for record in metadata["chunks"]]
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
    ids = [chunk.chunk_id for chunk in chunks]
    if (len(set(ids)) != len(ids) or bm25.corpus_size != len(chunks)
            or metadata.get("chunk_count") != len(chunks) or corpus["chunk_ids"] != ids
            or corpus.get("fingerprint") != metadata["fingerprint"]):
        raise ValueError("BM25 and metadata refer to different news chunks; rebuild the news index.")
    return bm25, chunks, corpus


# 질문을 동일한 Kiwi 정책으로 토큰화하고 BM25 점수순 청크 반환; 중요도 가산 없음
def search_news_keywords(
    question: str, *, top_k: int = 5, stock_code: str | None = None,
    start_date: str | None = None, end_date: str | None = None,
    min_score: float | None = None, bm25_path: Path = BM25_TOKENS_PATH,
    metadata_path: Path | None = None, app_settings: Settings = settings,
    kiwi: Kiwi | None = None,
) -> list[dict]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if min_score is not None and (type(min_score) not in (int, float) or not math.isfinite(min_score)):
        raise ValueError("BM25 min_score must be a finite number.")
    if stock_code is not None and (not isinstance(stock_code, str) or not stock_code.strip()):
        raise ValueError("stock_code must be a nonempty string.")
    start = date.fromisoformat(start_date) if start_date is not None else None
    end = date.fromisoformat(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must not be after end_date.")
    tokens = tokenize_text(question.strip(), kiwi=kiwi)
    if not tokens:
        return []
    destination = metadata_path if metadata_path is not None else (
        PROJECT_ROOT / app_settings.vector_db_path / "stock_advisor_metadata.json"
    )
    bm25, chunks, corpus = _load_keyword_index(bm25_path, destination)
    query_tokens = set(tokens)
    # 일치 토큰이 없는 문서는 제외하되 IDF에 따라 0·음수인 실제 일치 결과는 유지
    candidates = [
        position for position, chunk in enumerate(chunks)
        if (stock_code is None or chunk.stock_code == stock_code)
        and (start is None or date.fromisoformat(chunk.published_date) >= start)
        and (end is None or date.fromisoformat(chunk.published_date) <= end)
        and query_tokens.intersection(corpus["tokens"][position])
    ]
    if not candidates:
        return []
    scores = bm25.get_scores(tokens)
    if any(not math.isfinite(float(scores[position])) for position in candidates):
        raise ValueError("Invalid BM25 search scores.")
    candidates.sort(key=lambda position: (-float(scores[position]), chunks[position].chunk_id))
    return [
        {**asdict(chunks[position]), "score": float(scores[position])}
        for position in candidates
        if min_score is None or float(scores[position]) >= min_score
    ][:top_k]

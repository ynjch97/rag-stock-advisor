from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import httpx
from kiwipiepy import Kiwi
from openai import OpenAI

from src.chunking.document_chunker import chunk_and_save_news
from src.collectors.news_collector import collect_and_save_news
from src.config.paths import (BM25_TOKENS_PATH, NEWS_CHUNKS_DIR, PROCESSED_NEWS_DIR,
                              PROJECT_ROOT, RAW_NEWS_DIR, get_raw_news_path)
from src.config.settings import Settings, settings
from src.domain.advice_models import ParsedStock
from src.indexing.news_indexer import index_news
from src.preprocessing.news_preprocessor import preprocess_and_save_news
from src.services.execution_budget import deadline_http_client, remaining_timeout


KST = timezone(timedelta(hours=9))


# 지정 종목·기간·페이지 수집부터 인덱싱까지 조립하고 실제 신규 기사 수와 경로 반환
def collect_and_index_news(
    stock: ParsedStock, start_date: str, end_date: str, *, start: int = 1,
    app_settings: Settings = settings, deadline: float | None = None,
    raw_dir: Path = RAW_NEWS_DIR, processed_dir: Path = PROCESSED_NEWS_DIR,
    chunks_dir: Path = NEWS_CHUNKS_DIR, faiss_dir: Path | None = None,
    bm25_path: Path = BM25_TOKENS_PATH, client: OpenAI | None = None,
    http_client: httpx.Client | None = None, kiwi: Kiwi | None = None,
) -> dict:
    remaining_timeout(deadline)
    directory = faiss_dir if faiss_dir is not None else PROJECT_ROOT / app_settings.vector_db_path
    metadata_path = directory / "stock_advisor_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("chunks", []), list):
        raise ValueError("Invalid saved news index metadata.")
    old_chunks = metadata.get("chunks", [])
    old_urls = {item["url"] for item in old_chunks if item["stock_code"] == stock.stock_code}
    collected_at = datetime.now(KST)
    raw_path = raw_dir / get_raw_news_path(stock.stock_code, collected_at).name
    with (nullcontext(http_client) if http_client is not None else
          deadline_http_client(deadline, app_settings.agent_request_timeout_seconds)) as connection:
        payload = collect_and_save_news(stock.stock_name, stock.stock_code, start_date, end_date,
                                        start=start, app_settings=app_settings, http_client=connection,
                                        file_path=raw_path, now=collected_at)
    remaining_timeout(deadline)
    processed_at = datetime.now(KST)
    # 초 단위 news_id 규칙을 유지하면서 빠른 연속 페이지 처리의 ID 충돌 방지
    used_prefixes = {item["news_id"].split("_", 1)[0] for item in old_chunks}
    while processed_at.strftime("%y%m%d%H%M%S") in used_prefixes:
        time.sleep(min(0.05, remaining_timeout(deadline)))
        processed_at = datetime.now(KST)
    processed_path = processed_dir / raw_path.name
    records = preprocess_and_save_news(raw_path, file_path=processed_path, now=processed_at)
    remaining_timeout(deadline)
    chunk_path = chunks_dir / f"{processed_path.stem}_chunks.json"
    chunk_and_save_news(processed_path, file_path=chunk_path)
    result = index_news(
        chunks_dir, app_settings=app_settings, faiss_dir=directory, bm25_path=bm25_path,
        raw_news_dir=raw_dir, client=client, kiwi=kiwi, deadline=deadline,
        max_chunks=app_settings.agent_max_index_chunks,
        max_characters=app_settings.agent_max_index_characters,
    )
    eligible_urls = {record.url for record in records if start_date <= record.published_date <= end_date}
    return {**result, "collection": payload["collection"], "new_articles": len(eligible_urls - old_urls),
            "raw_path": str(raw_path), "processed_path": str(processed_path), "chunks_path": str(chunk_path)}

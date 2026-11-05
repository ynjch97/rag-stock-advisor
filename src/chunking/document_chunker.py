from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import re
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.paths import get_news_chunks_path
from src.domain.news_models import NewsChunk, ProcessedNews


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
FINANCIAL_NUMBER = re.compile(
    r"(?:[+\-−][ \t]*)?(?:[₩$€¥][ \t]*)?\d[\d,]*(?:\.\d+)?"
    r"(?:[ \t]*(?:조|억|만|천))?(?:[ \t]*(?:원|달러|엔|유로|주|배|명))?(?:[ \t]*%)?"
)


# 기사 본문을 분할하고 원문 위치·뉴스 ID·출처를 보존한 청크 모델 반환
def chunk_news(
    news: ProcessedNews, *, chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[NewsChunk]:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be an integer between 0 and chunk_size - 1.")
    if any(not isinstance(value, str) or not value.strip() for value in (
        news.news_id, news.content, news.stock_name, news.stock_code,
        news.title, news.publisher, news.published_date, news.url,
    )) or news.important not in ("O", "X"):
        raise ValueError("News must include nonempty content, ID, and metadata.")
    date.fromisoformat(news.published_date)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
        separators=[r"\n\n", r"\n", r"(?<=[.!?。！？])\s+", " ", ""],
        is_separator_regex=True, keep_separator="end", strip_whitespace=False, add_start_index=True,
    )
    spans = _split_content_spans(news.content, splitter)
    return [
        NewsChunk(
            chunk_id=f"{news.news_id}_{index:03d}", news_id=news.news_id,
            chunk_index=index, chunk_count=len(spans), start_index=start,
            content=news.content[start:end], stock_name=news.stock_name, stock_code=news.stock_code,
            title=news.title, important=news.important, publisher=news.publisher,
            published_date=news.published_date, url=news.url,
        )
        for index, (start, end) in enumerate(spans, start=1)
    ]


# 라이브러리 분할 위치가 금융 수치·단위 내부이면 경계를 넓혀 원문 표현을 온전히 보존
def _split_content_spans(
    content: str, splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[int, int]]:
    numbers = [match.span() for match in FINANCIAL_NUMBER.finditer(content)]
    spans = []
    for document in splitter.create_documents([content]):
        start = document.metadata["start_index"]
        if start < 0:
            raise ValueError("Cannot locate a chunk in the original news content.")
        end = start + len(document.page_content)
        for left, right in numbers:
            if left < start < right:
                start = left
            if left < end < right:
                end = right
        if (start, end) not in spans:
            spans.append((start, end))
    return spans


# processed 스냅샷의 기사들을 각각 청킹하고 대응하는 chunks JSON에 저장
def chunk_and_save_news(
    processed_path: Path, file_path: Path | None = None, *,
    chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[NewsChunk]:
    destination = file_path if file_path is not None else get_news_chunks_path(processed_path)
    if destination.resolve() == processed_path.resolve():
        raise ValueError("Chunk output must not overwrite processed news.")
    with processed_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Processed news must contain a JSON array.")
    chunks = []
    news_ids = set()
    for record in records:
        try:
            news = ProcessedNews(**record)
        except TypeError:
            raise ValueError("Invalid processed news record; regenerate outdated processed files.") from None
        article_chunks = chunk_news(news, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if news.news_id in news_ids:
            raise ValueError("Duplicate news_id in processed news.")
        chunks.extend(article_chunks)
        news_ids.add(news.news_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump([asdict(chunk) for chunk in chunks], file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return chunks

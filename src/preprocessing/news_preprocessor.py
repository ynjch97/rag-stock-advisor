from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.collectors.news_collector import NEWS_PUBLISHERS
from src.config.paths import get_processed_news_path
from src.domain.news_models import ProcessedNews
from src.preprocessing.text_cleaner import clean_news_text


KST = timezone(timedelta(hours=9))
TITLE_SIMILARITY_THRESHOLD = 0.90
BODY_SIMILARITY_THRESHOLD = 0.92
NUMBER_PATTERN = re.compile(r"[-+−]?\d+(?:[.,]\d+)*(?:\s*(?:%|조|억|만|원|달러|배))?")
NEGATION_PATTERN = re.compile(r"않|없|못|아니|불가|부인|철회")


# raw 기사에서 본문을 우선 사용하고 요약문 대체·정제·메타데이터 표준화·중복 제거 수행
def preprocess_news(raw_data: dict[str, Any]) -> list[ProcessedNews]:
    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("collection"), dict):
        raise ValueError("Raw news must include collection metadata.")
    metadata = raw_data["collection"]
    stock_name = clean_news_text(metadata.get("stock_name"))
    stock_code = str(metadata.get("stock_code", "")).strip().upper()
    if stock_code.isascii() and stock_code.isdigit():
        stock_code = stock_code.zfill(6)
    if not stock_name or re.fullmatch(r"[A-Z0-9]{6}", stock_code) is None:
        raise ValueError("Invalid news stock name or stock code.")
    try:
        start, end = date.fromisoformat(metadata["start_date"]), date.fromisoformat(metadata["end_date"])
        collected_at = datetime.fromisoformat(metadata["collected_at"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid news collection dates.") from None
    if collected_at.tzinfo is None or start > end:
        raise ValueError("News collection dates require an ordered range and timezone.")
    items = raw_data.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError("Raw news items must be an array of objects.")

    records = []
    for item in items:
        record = _normalize_news_item(item, stock_name, stock_code, collected_at)
        if record is not None:
            records.append(record)
    # 원문이 있는 최신 기사를 우선하여 중복 기사의 대표로 선택
    records.sort(key=lambda record: (record.important == "O", record.published_date), reverse=True)
    unique = []
    urls = set()
    for record in records:
        if record.url in urls or any(_is_similar_news(record, previous) for previous in unique):
            continue
        urls.add(record.url)
        unique.append(record)
    return unique


# 기사 한 건을 표준화하고 원문 본문 존재 여부로 중요도 설정
def _normalize_news_item(
    item: dict[str, Any], stock_name: str, stock_code: str,
    collected_at: datetime,
) -> ProcessedNews | None:
    original = item.get("original_article")
    original = original if isinstance(original, dict) else {}
    title = clean_news_text(item.get("title"))
    summary = clean_news_text(item.get("description"))
    original_body = original.get("body")
    has_original_body = isinstance(original_body, str) and bool(original_body.strip())
    body = clean_news_text(original_body)
    content = body or summary # 원문을 우선하고 없으면 요약문을 사용
    if not title or not content:
        return None
    raw_date = item.get("pubDate")
    if not isinstance(raw_date, str):
        return None
    try:
        published_at = parsedate_to_datetime(raw_date)
        if published_at.tzinfo is None or published_at > collected_at:
            return None
        published_date = published_at.astimezone(KST).date()
    except (TypeError, ValueError, OverflowError):
        return None
    url = _normalize_news_url(original.get("final_url") or item.get("originallink") or item.get("link"))
    if not url:
        return None
    host = urlsplit(url).hostname
    publisher = next((name for domain, name in NEWS_PUBLISHERS.items()
                      if host == domain or host.endswith("." + domain)), host)
    return ProcessedNews(
        stock_name=stock_name, stock_code=stock_code, url=url, title=title,
        content=content, publisher=publisher,
        published_date=published_date.isoformat(), important="O" if has_original_body else "X",
    )


# 기사 식별 쿼리는 유지하고 URL fragment와 알려진 추적 파라미터만 제거
def _normalize_news_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        address = urlsplit(value.strip())
        if (address.scheme not in ("http", "https") or not address.hostname
                or address.username or address.password or address.port not in (None, 80, 443)):
            return ""
    except ValueError:
        return ""
    query = [(key, val) for key, val in parse_qsl(address.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in ("fbclid", "gclid")]
    return urlunsplit((address.scheme, address.netloc.lower(), address.path or "/", urlencode(query), ""))


# 동일 발행일의 제목·본문이 모두 유사하고 수치·부정 표현도 같을 때만 중복으로 판단
def _is_similar_news(first: ProcessedNews, second: ProcessedNews) -> bool:
    if first.published_date != second.published_date:
        return False
    first_text, second_text = first.title + first.content, second.title + second.content
    if (NUMBER_PATTERN.findall(first_text) != NUMBER_PATTERN.findall(second_text)
            or NEGATION_PATTERN.findall(first_text) != NEGATION_PATTERN.findall(second_text)):
        return False
    title_score = SequenceMatcher(None, first.title.casefold(), second.title.casefold()).ratio()
    return title_score >= TITLE_SIMILARITY_THRESHOLD and SequenceMatcher(
        None, first.content.casefold(), second.content.casefold(),
    ).ratio() >= BODY_SIMILARITY_THRESHOLD


# raw 스냅샷 하나를 전처리하고 같은 이름의 processed 스냅샷에 원자적으로 저장
def preprocess_and_save_news(raw_path: Path, file_path: Path | None = None) -> list[ProcessedNews]:
    with raw_path.open("r", encoding="utf-8") as file:
        records = preprocess_news(json.load(file))
    destination = file_path if file_path is not None else get_processed_news_path(raw_path)
    if destination.resolve() == raw_path.resolve():
        raise ValueError("Processed output must not overwrite raw news.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump([asdict(record) for record in records], file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return records

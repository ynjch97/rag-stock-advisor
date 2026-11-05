from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.robotparser import RobotFileParser

import httpx
from trafilatura import extract

from src.config.paths import get_raw_news_path
from src.config.settings import Settings, settings


NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
NEWS_TIMEOUT_SECONDS = 10.0
NEWS_USER_AGENT = "RagStockAdvisor/1.0"
KST = timezone(timedelta(hours=9))
NEWS_PUBLISHERS = {
    "hankyung.com": "한국경제", "mk.co.kr": "매일경제", "sedaily.com": "서울경제", "biz.heraldcorp.com": "헤럴드경제", "economist.co.kr": "이코노미스트",
    "edaily.co.kr": "이데일리", "yna.co.kr": "연합뉴스", "mt.co.kr": "머니투데이", "biz.chosun.com": "조선비즈", "asiae.co.kr": "아시아경제",
}


# 종목별 최신 30건의 API 응답을 보존하고 기간·언론사 조건에 맞는 원문 조회 결과를 추가해 저장
def collect_and_save_news(
    stock_name: str, stock_code: str, start_date: str, end_date: str,
    *, app_settings: Settings = settings, http_client: httpx.Client | None = None,
    file_path: Path | None = None, now: datetime | None = None,
) -> dict[str, Any]:
    if not stock_name.strip() or re.fullmatch(r"[A-Za-z0-9]{6}", stock_code) is None:
        raise ValueError("A stock name and a six-character stock code are required.")
    checked_at = now if now is not None else datetime.now(KST)
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("Collection time must include a timezone offset.")
    checked_at = checked_at.astimezone(KST)
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end or end > checked_at.date():
        raise ValueError("Invalid or future news date range.")
    if (app_settings.news_api_provider or "").lower() != "naver":
        raise ValueError("News collection requires NEWS_API_PROVIDER=naver.")
    if not app_settings.news_api_key or not app_settings.news_api_secret:
        raise ValueError("Naver requires NEWS_API_KEY (Client ID) and NEWS_API_SECRET.")

    with (nullcontext(http_client) if http_client is not None
          else httpx.Client(timeout=NEWS_TIMEOUT_SECONDS)) as client:
        response = client.get(
            NAVER_NEWS_URL, params={"query": stock_name.strip(), "sort": "date", "display": 30, "start": 1},
            headers={"X-Naver-Client-Id": app_settings.news_api_key,
                     "X-Naver-Client-Secret": app_settings.news_api_secret},
            timeout=NEWS_TIMEOUT_SECONDS, follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        if (not isinstance(payload, dict) or not isinstance(payload.get("items"), list)
                or any(not isinstance(item, dict) for item in payload["items"])):
            raise ValueError("Invalid Naver news response: items must be an array of objects.")
        robots_cache: dict[str, RobotFileParser] = {}
        for item in payload["items"]:
            original = {"url": item.get("originallink"), "publisher": None,
                        "status": "pending", "html": None, "body": None, "error": None}
            item["original_article"] = original
            try:
                published_at = parsedate_to_datetime(item.get("pubDate", ""))
                if published_at.tzinfo is None:
                    raise ValueError("Missing news timezone.")
                published_at = published_at.astimezone(KST)
            except (TypeError, ValueError, OverflowError):
                original["status"] = "invalid_date"
                continue
            if not start <= published_at.date() <= end or published_at > checked_at:
                original["status"] = "outside_period"
                continue
            original["publisher"] = _get_publisher(original["url"])
            if original["publisher"] is None:
                original["status"] = "unsupported_publisher"
                continue
            _fetch_original_article(client, original, robots_cache)

    payload["collection"] = {
        "stock_name": stock_name.strip(), "stock_code": stock_code,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "collected_at": checked_at.isoformat(),
    }
    destination = file_path if file_path is not None else get_raw_news_path(stock_code, checked_at)
    _save_raw_news(destination, payload)
    return payload


# 지정 언론사의 HTTP(S) 주소만 허용하며 유사 도메인·인증정보·비표준 포트는 거부
def _get_publisher(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    try:
        address = httpx.URL(url)
        if address.scheme not in ("http", "https") or address.userinfo or address.port not in (None, 80, 443):
            return None
        return next((publisher for domain, publisher in NEWS_PUBLISHERS.items()
                     if address.host == domain or address.host.endswith("." + domain)), None)
    except httpx.InvalidURL:
        return None


# robots.txt 허용 여부를 호스트별로 한 번 조회하고 정책을 확인할 수 없으면 수집 중단
def _can_fetch_article(client: httpx.Client, url: str, cache: dict[str, RobotFileParser]) -> bool:
    address = httpx.URL(url)
    origin = str(address.copy_with(path="/robots.txt", query=None, fragment=None))
    if origin not in cache:
        response = client.get(origin, headers={"User-Agent": NEWS_USER_AGENT},
                              timeout=NEWS_TIMEOUT_SECONDS, follow_redirects=False)
        parser = RobotFileParser()
        if response.status_code == 404:
            parser.parse(["User-agent: *", "Disallow:"])
        elif response.status_code in (401, 403):
            parser.parse(["User-agent: *", "Disallow: /"])
        else:
            response.raise_for_status()
            parser.parse(response.text.splitlines())
        cache[origin] = parser
    return cache[origin].can_fetch(NEWS_USER_AGENT, url)


# 리다이렉트 대상도 검증하고 HTML·추출 본문·개별 실패 상태를 원본 응답에 기록
def _fetch_original_article(
    client: httpx.Client, result: dict[str, Any], robots_cache: dict[str, RobotFileParser],
) -> None:
    url = result["url"]
    try:
        for _ in range(6):
            if _get_publisher(url) is None:
                result["status"] = "unsupported_redirect"
                return
            if not _can_fetch_article(client, url, robots_cache):
                result["status"] = "robots_denied"
                return
            response = client.get(url, headers={"User-Agent": NEWS_USER_AGENT},
                                  timeout=NEWS_TIMEOUT_SECONDS, follow_redirects=False)
            if response.is_redirect:
                url = str(response.url.join(response.headers["location"]))
                continue
            response.raise_for_status()
            result["final_url"] = str(response.url)
            result["publisher"] = _get_publisher(str(response.url))
            result["html"] = response.text
            result["body"] = extract(response.content, url=str(response.url), include_comments=False)
            result["status"] = "success" if result["body"] else "empty_body"
            return
        result["status"] = "redirect_limit"
    except (httpx.HTTPError, KeyError, ValueError) as error:
        result["status"] = "fetch_failed"
        result["error"] = type(error).__name__


# 전체 응답을 임시 파일에 쓴 뒤 교체하여 저장 실패 시 기존 파일 보존
def _save_raw_news(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=file_path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

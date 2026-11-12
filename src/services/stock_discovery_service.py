from dataclasses import replace
from email.utils import parsedate_to_datetime
import re
import time

import httpx
from openai import OpenAI

from src.collectors.news_collector import KST, collect_theme_news
from src.config.settings import Settings, settings
from src.data_loaders.stock_reference_loader import load_stock_references
from src.domain.advice_models import (MAX_DISCOVERY_RESULTS, StockDiscoveryCandidate,
                                      StockDiscoveryCriteria, StockDiscoveryEvidence)
from src.parsers.llm_stock_discovery_analyzer import (
    MAX_DISCOVERY_ARTICLE_CHARACTERS, analyze_discovery_candidates,
)
from src.preprocessing.text_cleaner import clean_news_text
from src.services.execution_budget import remaining_timeout


# 기준정보의 정식 종목명 매칭에 공백 제거·대소문자 정규화 적용
def _normalize_stock_name(value: str) -> str:
    return "".join(value.split()).casefold()


# 성공적으로 확보한 원문만 출처 단위로 정리; 요약만 있는 기사를 사업 검증 근거로 확정하지 않음
def _prepare_discovery_articles(payload: dict, deadline: float) -> list[StockDiscoveryEvidence]:
    articles, seen = [], set()
    for item in payload["items"]:
        remaining_timeout(deadline)
        original = item.get("original_article")
        if not isinstance(original, dict) or original.get("status") != "success":
            continue
        content = clean_news_text(original.get("body"))
        title = clean_news_text(item.get("title"))
        if not content or not title:
            continue
        try:
            address = httpx.URL(original.get("final_url") or item.get("originallink"))
            published = parsedate_to_datetime(item.get("pubDate", ""))
        except (TypeError, ValueError, httpx.InvalidURL):
            continue
        if (address.scheme not in ("http", "https") or not address.host or address.userinfo
                or address.port not in (None, 80, 443) or published.utcoffset() is None):
            continue
        url = str(address.copy_with(fragment=None))
        if url in seen:
            continue
        seen.add(url)
        articles.append(StockDiscoveryEvidence(title=title, url=url, content=content,
            publisher=original.get("publisher") or address.host, published_date=published.astimezone(KST).date().isoformat()))
    return articles


# 테마 뉴스에서 검토 후보를 찾고 상장 종목·원문 인용·직접 사업 연관성을 확인한 결과만 반환
def discover_stocks(
    criteria: StockDiscoveryCriteria, *, start_date: str | None = None, end_date: str | None = None,
    app_settings: Settings = settings, stock_infos: list[dict[str, str]] | None = None,
    client: OpenAI | None = None, http_client: httpx.Client | None = None, deadline: float | None = None,
) -> list[StockDiscoveryCandidate]:
    if (not isinstance(criteria.theme, str) or not criteria.theme.strip() or len(criteria.theme.strip()) > 500
            or type(criteria.requested_count) is not int or not 1 <= criteria.requested_count <= MAX_DISCOVERY_RESULTS):
        raise ValueError("Discovery requires a theme and requested_count between 1 and 10.")
    limit = app_settings.agent_max_discovery_candidates
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("AGENT_MAX_DISCOVERY_CANDIDATES must be between 1 and 10.")
    deadline = deadline if deadline is not None else time.monotonic() + app_settings.agent_timeout_seconds
    remaining_timeout(deadline)
    if client is None and not app_settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for stock discovery.")
    records = stock_infos if stock_infos is not None else load_stock_references()
    if not isinstance(records, list):
        raise ValueError("Stock references must contain a list of records.")
    by_name = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid stock reference record for discovery.")
        name, code = record.get("stock_name"), record.get("stock_code")
        if not isinstance(name, str) or not name.strip() or not isinstance(code, str):
            raise ValueError("Invalid stock reference fields for discovery.")
        if re.fullmatch(r"[A-Za-z0-9]{6}", code):
            by_name.setdefault(_normalize_stock_name(name), {})[code] = record
    if not by_name:
        return []
    payload = collect_theme_news(criteria.theme.strip(), start_date, end_date,
        app_settings=app_settings, http_client=http_client, start=1, deadline=deadline)
    articles = _prepare_discovery_articles(payload, deadline)
    # 1. 후보 추출에 사용할 기사 부재를 확인 (분석 가능한 원문이 없으면 true, 빈 후보 목록 반환)
    if not articles:
        return []
    analysis = analyze_discovery_candidates(criteria, articles, limit,
        app_settings=app_settings, client=client, deadline=deadline)
    remaining_timeout(deadline)
    candidates = {}
    for item in analysis.candidates:
        matches = by_name.get(_normalize_stock_name(item.stock_name), {})
        # 2. 후보의 기준정보 일치·선정 이유를 확인 (일치 코드가 정확히 1개가 아니거나 선정 이유가 없으면 true, 후보 제외)
        if len(matches) != 1 or not item.selection_reason.strip():
            continue
        record = next(iter(matches.values()))
        evidence, urls = [], set()
        for reference in item.evidence:
            if (
                # 3. 후보의 직접 사업 연관성을 확인 (LLM이 분류한 관계가 direct가 아니면 true, 근거 제외)
                reference.relation != "direct"
                # 4. 근거의 출처 번호를 확인 (전달한 기사 목록의 범위를 초과하면 true, 근거 제외)
                or reference.source_index >= len(articles)
            ):
                continue
            article = articles[reference.source_index]
            excerpt = " ".join(reference.excerpt.split())
            visible_content = " ".join(article.content[:MAX_DISCOVERY_ARTICLE_CHARACTERS].split())
            if (
                # 5. 근거의 인용문 누락을 확인 (인용문이 비어 있으면 true, 근거 제외)
                not excerpt
                # 6. 실제 원문 인용 여부를 확인 (인용문이 LLM에 제공한 본문 앞 2,000자에 없으면 true, 근거 제외)
                or excerpt not in visible_content
                # 7. 후보 내 근거 중복을 확인 (동일 URL을 이미 추가했으면 true, 중복 근거 제외)
                or article.url in urls
            ):
                continue
            urls.add(article.url)
            evidence.append(replace(article, excerpt=excerpt))
        # 8. 후보의 검증된 근거 부재를 확인 (통과한 근거가 없으면 true, 후보 제외)
        if not evidence:
            continue
        code = record["stock_code"]
        # 9. 후보 종목 중복을 확인 (동일 종목코드가 이미 있으면 true, 기존 후보에 근거만 병합)
        if code in candidates:
            previous = candidates[code]
            previous_urls = {source.url for source in previous.evidence}
            merged = previous.evidence + tuple(source for source in evidence if source.url not in previous_urls)
            candidates[code] = replace(previous, evidence=merged[:3])
        else:
            candidates[code] = StockDiscoveryCandidate(stock_name=record["stock_name"], stock_code=code,
                selection_reason=item.selection_reason.strip(), evidence=tuple(evidence), business_verified=True)
    # 요청 개수는 최종 선택 단계에서 적용; 여기서는 최대 검토 후보 수만 제한
    return list(candidates.values())[:limit]

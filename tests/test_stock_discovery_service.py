from copy import deepcopy
from dataclasses import replace
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.collectors import news_collector
from src.config.settings import settings
from src.domain.advice_models import StockDiscoveryCriteria
from src.parsers.llm_stock_discovery_analyzer import DiscoveryAnalysis
from src.services import stock_discovery_service as service


QUOTE = "소재기업A는 반도체 소재를 생산한다."
REFERENCES = [{"stock_name": "소재기업A", "stock_code": "000001", "market": "KOSPI"}]


# 원문 추출 성공·발행 시각·출처를 갖춘 수집 응답
@pytest.fixture
def payload():
    return {"collection": {"query": "반도체 소재 관련주", "start": 1, "next_start": None},
        "items": [{"title": "소재기업A 사업", "originallink": "https://www.hankyung.com/article/1",
        "description": "요약", "pubDate": "Thu, 05 Nov 2026 12:00:00 +0900",
        "original_article": {"status": "success", "body": QUOTE + " 추가 상세 본문.",
                             "publisher": "한국경제"}}]}


# 후보 추출 응답을 strict 모델로 검증한 뒤 제공
def _analysis(name="소재기업A", excerpt=QUOTE, relation="direct", source_index=0):
    return DiscoveryAnalysis.model_validate({"candidates": [{"stock_name": name,
        "selection_reason": "반도체 소재 생산 사업", "evidence": [{"source_index": source_index,
        "excerpt": excerpt, "relation": relation}]}]})


# 수집과 LLM만 대체하고 실제 종목·인용 검증 로직을 실행
@pytest.fixture
def dependencies(monkeypatch, payload):
    collect = Mock(return_value=payload)
    analyze = Mock(return_value=_analysis())
    monkeypatch.setattr(service, "collect_theme_news", collect)
    monkeypatch.setattr(service, "analyze_discovery_candidates", analyze)
    return collect, analyze


# 테스트마다 외부 인증정보를 사용하지 않는 공통 진입점
def _discover(**kwargs):
    options = {"stock_infos": REFERENCES, "client": Mock()}
    options.update(kwargs)
    return service.discover_stocks(StockDiscoveryCriteria("반도체 소재", 3), **options)["candidates"]


# 기준정보 코드와 원문 전체·인용·출처를 보존하며 최종 개수와 검토 상한을 분리
def test_verified_candidate(dependencies):
    collect, analyze = dependencies
    result = _discover(start_date="2026-11-01", end_date="2026-11-05")
    assert len(result) == 1
    candidate = result[0]
    assert candidate.stock_code == "000001"
    assert candidate.business_verified is True
    assert candidate.evidence[0].excerpt == QUOTE
    assert candidate.evidence[0].content == QUOTE + " 추가 상세 본문."
    assert candidate.evidence[0].published_date == "2026-11-05"
    assert candidate.evidence[0].publisher == "한국경제"
    assert analyze.call_args.args[2] == 10
    assert collect.call_args.args == ("반도체 소재 관련주", "2026-11-01", "2026-11-05")
    assert collect.call_args.kwargs["start"] == 1
    collect.assert_called_once()


# 임의 회사·부정/단순 언급·조작된 출처와 인용을 제외
@pytest.mark.parametrize("changes", [
    {"name": "없는기업"},
    {"excerpt": "소재기업A는 생산하지 않는다."}, {"source_index": 30},
    {"relation": "incidental"}, {"relation": "negated"}, {"relation": "uncertain"},
])
def test_invalid_business_evidence(dependencies, changes):
    dependencies[1].return_value = _analysis(**changes)
    assert _discover() == []


# 기업명이 인용문에 없어도 실제 원문과 직접 사업 연관성이 확인되면 허용
def test_excerpt_without_company_name(dependencies):
    excerpt = "반도체 소재를 생산한다."
    dependencies[1].return_value = _analysis(excerpt=excerpt)
    result = _discover()
    assert len(result) == 1
    assert result[0].stock_code == "000001"
    assert result[0].evidence[0].excerpt == excerpt


# 같은 이름에 복수 코드가 있으면 임의 코드 선택 금지
def test_ambiguous_reference(dependencies):
    assert not _discover(stock_infos=REFERENCES + [{"stock_name": "소재기업A", "stock_code": "000002"}])


# 중복 회사와 동일 출처는 한 번만 반환
def test_duplicate_candidates_and_articles(dependencies, payload):
    payload["items"].append(deepcopy(payload["items"][0]))
    dependencies[1].return_value = DiscoveryAnalysis(candidates=_analysis().candidates * 2)
    result = _discover()
    assert len(result) == 1
    assert len(result[0].evidence) == 1
    assert len(dependencies[1].call_args.args[1]) == 1


# 최종 요청 3개라도 검토 후보는 최대 10개까지 확보
def test_review_candidates_not_final_count(dependencies, payload):
    records, candidates = [], []
    payload["items"] = []
    for index in range(4):
        name = f"소재기업{index}"
        quote = f"{name}는 반도체 소재를 생산한다."
        records.append({"stock_name": name, "stock_code": f"{index + 1:06}"})
        payload["items"].append({"title": name, "originallink": f"https://example.com/{index}",
            "pubDate": "Thu, 05 Nov 2026 12:00:00 +0900",
            "original_article": {"status": "success", "body": quote}})
        candidates.extend(_analysis(name=name, excerpt=quote, source_index=index).candidates)
    dependencies[1].return_value = DiscoveryAnalysis(candidates=candidates)
    assert len(_discover(stock_infos=records)) == 4
    assert len(_discover(stock_infos=records,
                         app_settings=replace(settings, agent_max_discovery_candidates=2))) == 2


# 요약만 있거나 원문·URL·날짜가 잘못되면 사업 근거로 사용하지 않음
@pytest.mark.parametrize("field,value", [
    ("original_article", {"status": "unsupported_publisher", "body": QUOTE}),
    ("original_article", {"status": "success", "body": ""}),
    ("originallink", "javascript:alert(1)"), ("pubDate", "invalid"), ("title", ""),
])
def test_unusable_articles_skip_llm(dependencies, payload, field, value):
    payload["items"][0][field] = value
    assert not _discover()
    dependencies[1].assert_not_called()


# LLM에 전달하지 않은 본문 뒤쪽의 문장은 인용 근거로 인정하지 않음
def test_excerpt_outside_visible_text(dependencies, payload):
    payload["items"][0]["original_article"]["body"] = "다른 설명. " * 500 + QUOTE
    assert not _discover()


# 시간 예산·자격정보·기준정보가 부족하면 검색을 시작하지 않음
def test_preflight_validation(dependencies):
    with pytest.raises(TimeoutError):
        _discover(deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _discover(client=None, app_settings=replace(settings, openai_api_key=None))
    assert _discover(stock_infos=[]) == []
    dependencies[0].assert_not_called()


# 잘못된 테마·개수·후보 상한은 요청 전에 거부
@pytest.mark.parametrize("theme,count,limit", [("", 3, 10), ("소재", 11, 10),
                                               ("소재", True, 10), ("소재", 3, 11)])
def test_invalid_criteria(dependencies, theme, count, limit):
    with pytest.raises(ValueError):
        service.discover_stocks(StockDiscoveryCriteria(theme, count), stock_infos=REFERENCES, client=Mock(),
                               app_settings=replace(settings, agent_max_discovery_candidates=limit))
    dependencies[0].assert_not_called()


# 실제 수집·Tool Calling·코드 검증을 모의 통신만으로 연결하고 파일은 저장하지 않음
def test_mocked_end_to_end(monkeypatch, payload):
    monkeypatch.setattr(news_collector, "extract", Mock(return_value=QUOTE))
    monkeypatch.setattr(news_collector, "_save_raw_news", Mock(side_effect=AssertionError("No raw save")))
    connection = Mock()
    connection.with_options.return_value = connection
    connection.responses.create.return_value = SimpleNamespace(status="completed", output=[
        SimpleNamespace(type="function_call", name="submit_discovery_candidates",
                        arguments=_analysis().model_dump_json())])

    # 검색 응답·robots.txt·기사 HTML만 제공
    def respond(request):
        if request.url.host == "openapi.naver.com":
            assert request.url.params["query"] == "반도체 소재 관련주"
            return httpx.Response(200, json=payload)
        text = "User-agent: *\nDisallow:\n" if request.url.path == "/robots.txt" else "<article>본문</article>"
        return httpx.Response(200, text=text)

    config = replace(settings, news_api_provider="naver", news_api_key="test", news_api_secret="test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        result = _discover(client=connection, http_client=http_client, app_settings=config)
    assert result[0].stock_code == "000001"
    assert result[0].evidence[0].excerpt == QUOTE
    connection.responses.create.assert_called_once()


# 검색어에 관련주를 한 번만 추가하고 요청된 페이지를 그대로 전달
@pytest.mark.parametrize("theme,expected", [
    ("반도체 소재", "반도체 소재 관련주"),
    (" 반도체 소재 관련주 ", "반도체 소재 관련주"),
    ("반도체관련주 소재", "반도체관련주 소재"),
])
def test_discovery_query_and_second_page(dependencies, payload, theme, expected):
    payload["collection"] = {"query": expected, "start": 31, "next_start": None, "returned_count": 1}
    result = service.discover_stocks(StockDiscoveryCriteria(theme, 3), start=31,
        stock_infos=REFERENCES, client=Mock())
    assert dependencies[0].call_args.args[0] == expected
    assert dependencies[0].call_args.kwargs["start"] == 31
    assert len(result["candidates"]) == 1
    assert result["collection"] == payload["collection"]
    assert dependencies[1].call_args.args[0].theme == theme


# 원문 미확보·코드 불일치·LLM 후보 없음도 다음 페이지 정보를 유실하지 않음
@pytest.mark.parametrize("empty_stage", ["body", "mapping", "llm"])
def test_empty_discovery_preserves_next_page(dependencies, payload, empty_stage):
    payload["collection"] = {"query": "반도체 소재 관련주", "start": 1,
                             "next_start": 31, "returned_count": 30}
    if empty_stage == "body":
        payload["items"][0]["original_article"]["status"] = "unsupported_publisher"
    elif empty_stage == "mapping":
        dependencies[1].return_value = _analysis(name="없는기업")
    else:
        dependencies[1].return_value = DiscoveryAnalysis(candidates=[])
    result = service.discover_stocks(StockDiscoveryCriteria("반도체 소재", 3),
                                    stock_infos=REFERENCES, client=Mock())
    assert result["candidates"] == []
    assert result["collection"]["next_start"] == 31
    dependencies[0].assert_called_once()
    if empty_stage == "body":
        dependencies[1].assert_not_called()


# 지원하지 않는 페이지는 네이버·LLM 요청 전에 차단
@pytest.mark.parametrize("start", [0, 2, 61, True, "31"])
def test_discovery_rejects_invalid_start(dependencies, start):
    with pytest.raises(ValueError, match="start"):
        _discover(start=start)
    dependencies[0].assert_not_called()
    dependencies[1].assert_not_called()

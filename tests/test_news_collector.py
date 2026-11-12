from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
from unittest.mock import Mock

import httpx
import pytest

from src.collectors import news_collector as collector
from src.config.settings import settings


# 실제 인증정보·외부 통신 없이 수집 요청을 실행하는 설정 제공
@pytest.fixture
def config():
    return replace(settings, openai_api_key=None, stock_api_key=None, stock_api_secret=None,
                   news_api_provider="naver", news_api_key="test-client-id",
                   news_api_secret="test-client-secret")


# 네이버 응답의 원래 필드와 확장 필드 보존 여부를 비교할 모의 기사
@pytest.fixture
def payload():
    return {
        "lastBuildDate": "Thu, 05 Nov 2026 12:00:00 +0900", "total": 100, "start": 1, "display": 1,
        "items": [{"title": "<b>삼성전자</b> 실적 전망", "originallink": "https://www.hankyung.com/article/1",
                   "link": "https://n.news.naver.com/article/015/1", "description": "API <b>요약</b>",
                   "pubDate": "Sat, 31 Oct 2026 11:00:00 +0900", "extra": "preserve"}],
    }


# 시간을 고정하고 임시 파일에만 저장하는 수집 호출
def _collect(config, client, path):
    return collector.collect_and_save_news(
        "삼성전자", "005930", "2026-10-30", "2026-11-05",
        app_settings=config, http_client=client, file_path=path,
        now=datetime.fromisoformat("2026-11-05T12:00:00+09:00"),
    )


# 추가 페이지는 31부터 30건 조회하며 수집 이력에도 시작 위치 기록
def test_collect_second_page(tmp_path, config):
    # 추가 조회의 페이지 매개변수 검증
    def respond(request):
        assert request.url.params["start"] == "31"
        assert request.url.params["display"] == "30"
        return httpx.Response(200, json={"items": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = collector.collect_and_save_news("삼성전자", "005930", "2026-10-30", "2026-11-05",
            start=31, app_settings=config, http_client=client, file_path=tmp_path / "news.json",
            now=datetime.fromisoformat("2026-11-05T12:00:00+09:00"))
    assert result["collection"]["start"] == 31


# 허용된 두 페이지 외 시작 위치는 요청 전에 거부
@pytest.mark.parametrize("start", [0, 2, 61, True, "31"])
def test_collect_invalid_page(config, start):
    with pytest.raises(ValueError, match="News start"):
        collector.collect_and_save_news("삼성전자", "005930", "2026-10-30", "2026-11-05",
                                        start=start, app_settings=config)


# API 응답은 그대로 보존하며 주말 기사의 원문 조회 결과를 추가하고 저장
def test_collect_preserves_api_response_and_adds_original(tmp_path, config, payload, monkeypatch):
    source = deepcopy(payload)
    requests = []
    extractor = Mock(return_value="원문 본문: 영업이익 10조원, 감소하지 않았다.")
    monkeypatch.setattr(collector, "extract", extractor)

    # 네이버 인증정보는 검색 요청에만 전달되며 언론사에는 전달되지 않음
    def respond(request):
        requests.append(request)
        if request.url.host == "openapi.naver.com":
            assert dict(request.url.params) == {"query": "삼성전자", "sort": "date", "display": "30", "start": "1"}
            assert request.headers["X-Naver-Client-Id"] == "test-client-id"
            assert request.headers["X-Naver-Client-Secret"] == "test-client-secret"
            return httpx.Response(200, json=payload)
        assert "X-Naver-Client-Id" not in request.headers
        assert "X-Naver-Client-Secret" not in request.headers
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow:\n")
        return httpx.Response(200, text="<html><article>원문 HTML</article></html>")

    path = tmp_path / "raw" / "news.json"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _collect(config, client, path)
        assert not client.is_closed
    assert json.loads(path.read_text(encoding="utf-8")) == result
    assert {key: result[key] for key in source if key != "items"} == {key: value for key, value in source.items() if key != "items"}
    assert {key: result["items"][0][key] for key in source["items"][0]} == source["items"][0]
    original = result["items"][0]["original_article"]
    assert original["status"] == "success"
    assert original["publisher"] == "한국경제"
    assert original["html"] == "<html><article>원문 HTML</article></html>"
    assert original["body"] == extractor.return_value
    assert result["collection"]["stock_code"] == "005930"
    assert len(requests) == 3
    assert not (tmp_path / "processed").exists()


# 기간 밖·미래 시각·잘못된 날짜·대상 외 도메인은 응답을 보존하되 원문 요청 생략
@pytest.mark.parametrize("field,value,status", [
    ("pubDate", "Thu, 29 Oct 2026 11:00:00 +0900", "outside_period"),
    ("pubDate", "Thu, 05 Nov 2026 13:00:00 +0900", "outside_period"),
    ("pubDate", "not-a-date", "invalid_date"),
    ("originallink", "https://www.hankyung.com.evil.test/a", "unsupported_publisher"),
    ("originallink", "https://hankyung.com@127.0.0.1/a", "unsupported_publisher"),
    ("originallink", "http://127.0.0.1/a", "unsupported_publisher"),
    ("originallink", "https://www.hankyung.com:9999/a", "unsupported_publisher"),
])
def test_ineligible_items_are_kept_without_fetching(tmp_path, config, payload, field, value, status):
    payload["items"][0][field] = value
    requests = []

    # 검색 응답 외의 HTTP 호출은 허용하지 않음
    def respond(request):
        requests.append(request)
        assert request.url.host == "openapi.naver.com"
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _collect(config, client, tmp_path / "news.json")
    assert len(result["items"]) == 1
    assert result["items"][0]["original_article"]["status"] == status
    assert len(requests) == 1


# 실패한 원문 한 건 때문에 전체 API 응답을 잃거나 요약문으로 본문을 대체하지 않음
@pytest.mark.parametrize("failure,expected", [
    ("robots", "robots_denied"), ("http", "fetch_failed"), ("timeout", "fetch_failed"),
    ("empty", "empty_body"), ("redirect", "unsupported_redirect"),
])
def test_original_failures_are_saved(tmp_path, config, payload, monkeypatch, failure, expected):
    monkeypatch.setattr(collector, "extract", Mock(return_value=None))

    # 사이트 차단·접속 실패·본문 없음·대상 외 리다이렉트를 모의
    def respond(request):
        if request.url.host == "openapi.naver.com":
            return httpx.Response(200, json=payload)
        assert request.url.host == "www.hankyung.com"
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /" if failure == "robots" else "User-agent: *\nDisallow:")
        if failure == "http":
            return httpx.Response(403)
        if failure == "timeout":
            raise httpx.ReadTimeout("test", request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        return httpx.Response(200, text="<html></html>")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _collect(config, client, tmp_path / "news.json")
    original = result["items"][0]["original_article"]
    assert original["status"] == expected
    assert original["body"] is None
    assert result["items"][0]["description"] == "API <b>요약</b>"


# 같은 언론사의 robots 정책은 한 번만 조회하고 허용된 리다이렉트도 검증
def test_robots_cache_and_allowed_redirect(tmp_path, config, payload, monkeypatch):
    payload["items"].append({**payload["items"][0], "originallink": "https://www.hankyung.com/article/2"})
    monkeypatch.setattr(collector, "extract", Mock(return_value="본문"))
    robots_calls = []

    # 첫 기사만 같은 언론사의 다른 페이지로 이동
    def respond(request):
        if request.url.host == "openapi.naver.com":
            return httpx.Response(200, json=payload)
        if request.url.path == "/robots.txt":
            robots_calls.append(request.url)
            return httpx.Response(404)
        if request.url.path == "/article/1":
            return httpx.Response(302, headers={"Location": "/article/final"})
        return httpx.Response(200, text="<html>본문</html>")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _collect(config, client, tmp_path / "news.json")
    assert len(robots_calls) == 1
    assert all(item["original_article"]["status"] == "success" for item in result["items"])
    assert result["items"][0]["original_article"]["final_url"].endswith("/article/final")


# 검색 실패·잘못된 응답은 기존 raw를 덮어쓰지 않음
@pytest.mark.parametrize("failure", ["http", "shape"])
def test_search_failure_preserves_file(tmp_path, config, failure):
    path = tmp_path / "news.json"
    path.write_text("existing", encoding="utf-8")
    transport = httpx.MockTransport(lambda request: (
        httpx.Response(429, json={"errorMessage": "limited"}) if failure == "http"
        else httpx.Response(200, json={"items": {}})
    ))
    with httpx.Client(transport=transport) as client:
        with pytest.raises((httpx.HTTPStatusError, ValueError)):
            _collect(config, client, path)
    assert path.read_text(encoding="utf-8") == "existing"


# 키 누락·미지원 제공자·잘못된 종목·기간은 외부 호출 전에 실패
@pytest.mark.parametrize("failure", ["key", "secret", "provider", "code", "range", "future", "naive"])
def test_invalid_inputs_do_not_call_api(tmp_path, config, failure):
    if failure in ("key", "secret"):
        config = replace(config, **{f"news_api_{failure}": None})
    if failure == "provider":
        config = replace(config, news_api_provider="other")
    with httpx.Client(transport=httpx.MockTransport(Mock(side_effect=AssertionError("No API calls")))) as client:
        with pytest.raises(ValueError):
            collector.collect_and_save_news(
                "삼성전자", "../bad" if failure == "code" else "005930",
                "2026-11-06" if failure == "range" else "2026-10-30",
                "2026-11-06" if failure == "future" else "2026-11-05",
                app_settings=config, http_client=client, file_path=tmp_path / "news.json",
                now=datetime.fromisoformat("2026-11-05T12:00:00" + ("" if failure == "naive" else "+09:00")),
            )


# 실제 라이브러리로 본문을 추출하며 금융 수치와 부정 표현 보존
def test_real_body_extractor_preserves_article_content():
    html = """<html><head><title>실적 전망</title></head><body><article>
    <h1>삼성전자 실적 전망</h1>
    <p>삼성전자의 영업이익은 10조원으로 예상됐다. 매출은 전년보다 감소하지 않았다.</p>
    <p>시장에서는 반도체 수요가 실적에 영향을 줄 것으로 봤다. 회사는 투자 계획을 검토하고 있다.</p>
    </article></body></html>"""
    body = collector.extract(html, include_comments=False)
    assert body is not None
    assert "10조원" in body and "감소하지 않았다" in body

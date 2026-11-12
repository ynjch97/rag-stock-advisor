from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import json
import time
from unittest.mock import Mock

import httpx
import pytest

from src.app import stock_advice_workflow as workflow
from src.config.settings import settings
from src.domain.advice_models import AgentDecision, ParsedQuestion, ParsedStock
from src.services.stock_context_service import build_stock_context


# 초기 근거·실행 도구를 모두 메모리 대역으로 준비하여 외부 호출 차단
@pytest.fixture
def environment(monkeypatch):
    stock = ParsedStock("하이닉스", "SK하이닉스", "000660")
    parsed = ParsedQuestion("하이닉스 전망", (stock,), "general_analysis")
    context = build_stock_context(parsed, {"stock_name": stock.stock_name, "stock_code": stock.stock_code,
        "base_date": "2026-11-12", "current_price": 100, "trend": "상승", "source": "toss"}, [],
        news_source="hybrid_search")
    context.update(news_period={"start_date": "2026-11-06", "end_date": "2026-11-12"}, news_collections=[],
                   checked_at=datetime.now(workflow.KST).isoformat(),
                   availability={"stock": True, "stock_retryable": True, "news_retryable": True})
    collector, search, agent = Mock(), Mock(return_value=[]), Mock(return_value=AgentDecision("respond", "충분"))
    monkeypatch.setattr(workflow, "collect_and_index_news", collector)
    monkeypatch.setattr(workflow, "search_news_hybrid", search)
    monkeypatch.setattr(workflow, "decide_next_action", agent)
    monkeypatch.setattr(workflow, "_read_news_collections", Mock(return_value=[]))
    monkeypatch.setattr(workflow, "_missing_news_index", Mock(return_value=False))
    return parsed, [context], collector, search, agent


# 테스트용 서로 다른 기사 근거 제공
def _news(number=1):
    return [{"url": f"https://example.test/{number}", "content": f"근거 {number}",
             "title": f"기사 {number}", "published_date": "2026-11-12"}]


# 기본 예산으로 Controller 반복 실행
def _run(environment, config=settings, deadline=None):
    workflow._run_agent_loop(environment[0], environment[1], config,
                             time.monotonic() + 60 if deadline is None else deadline)


# 충분한 근거면 추가 수집 없이 즉시 응답
def test_sufficient_evidence_skips_collection(environment):
    environment[1][0]["news"] = _news()
    _run(environment)
    environment[2].assert_not_called()
    assert environment[1][0]["agent_history"][0]["action"] == "respond"


# 부족한 종목의 뉴스만 수집·재검색하고 보충된 결과를 다음 판단에 전달
def test_missing_news_is_collected_and_researched(environment):
    parsed, contexts, collector, search, agent = environment
    collector.return_value = {"new_articles": 1}
    search.return_value = _news()
    agent.side_effect = [AgentDecision("collect_news", "뉴스 없음", parsed.stocks[0], news_start=1),
                         AgentDecision("respond", "근거 보충 완료")]
    _run(environment)
    assert collector.call_count == search.call_count == 1
    assert contexts[0]["summary"]["news_count"] == 1
    assert agent.call_count == 2
    assert contexts[0]["agent_history"][0]["new_articles"] == 1


# 페이지 1·31만 각 한 번 허용
def test_two_pages_are_collected_once(environment):
    parsed, _, collector, search, agent = environment
    stock = parsed.stocks[0]
    collector.return_value = {"new_articles": 1}
    search.side_effect = [_news(1), _news(2)]
    agent.side_effect = [AgentDecision("collect_news", "첫 페이지", stock, news_start=1),
                         AgentDecision("collect_news", "추가 페이지", stock, news_start=31),
                         AgentDecision("respond", "충분")]
    _run(environment)
    assert [call.kwargs["start"] for call in collector.call_args_list] == [1, 31]
    assert agent.call_count == 3


# LLM이 동일 페이지 재요청해도 서버 예산 검증에서 차단
def test_duplicate_page_is_rejected(environment):
    parsed, contexts, collector, search, agent = environment
    collector.return_value, search.return_value = {"new_articles": 1}, _news()
    agent.return_value = AgentDecision("collect_news", "수집", parsed.stocks[0], news_start=1)
    _run(environment)
    assert collector.call_count == 1
    assert any("Agent 처리 실패" in note for note in contexts[0]["issues"])


# 최소 주가·뉴스 근거가 없는 상태에서는 충분함 결정을 허용하지 않음
def test_respond_requires_minimum_evidence(environment):
    parsed, contexts, _, _, _ = environment
    allowed = workflow._allowed_agent_actions(parsed, contexts, [], 0, settings)
    assert not any(item["action"] == "respond" for item in allowed)
    contexts[0]["news"] = _news()
    assert any(item["action"] == "respond" for item in workflow._allowed_agent_actions(parsed, contexts, [], 0, settings))


# 손상 파일과 정상적인 빈 수집 이력 상태 구분
def test_missing_index_distinguishes_corruption(tmp_path):
    config = replace(settings, vector_db_path=str(tmp_path))
    metadata_path = tmp_path / "stock_advisor_metadata.json"
    metadata_path.write_text("invalid json", encoding="utf-8")
    assert workflow._missing_news_index(config) is False
    metadata_path.write_text(json.dumps({"chunk_count": 0}), encoding="utf-8")
    assert workflow._missing_news_index(config) is True
    (tmp_path / "stock_advisor.index").write_bytes(b"stale index")
    assert workflow._missing_news_index(config) is False


# 잘못된 수집 이력을 최신 정보처럼 사용하지 않음
def test_invalid_collection_history_is_rejected(tmp_path):
    config = replace(settings, vector_db_path=str(tmp_path))
    path = tmp_path / "stock_advisor_metadata.json"
    path.write_text(json.dumps({"collections": [{"stock_code": "000660"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="collection fields"):
        workflow._read_news_collections("000660", config)


# 기존 메타데이터도 대응 raw에서 페이지 상태를 복원하며 원본 파일은 변경하지 않음
def test_legacy_collection_restores_page_status(tmp_path, monkeypatch):
    name = "000660_news_20261113_030000_000001"
    collection = {"stock_code": "000660", "start_date": "2026-11-06", "end_date": "2026-11-12",
                  "collected_at": datetime.now(workflow.KST).isoformat(), "start": 1,
                  "article_count": 0, "chunk_file": name + "_chunks.json"}
    metadata = tmp_path / "stock_advisor_metadata.json"
    metadata.write_text(json.dumps({"collections": [collection]}), encoding="utf-8")
    before = metadata.read_bytes()
    (tmp_path / (name + ".json")).write_text(json.dumps({"items": [
        {"pubDate": "Fri, 13 Nov 2026 02:00:00 +0900"}] * 30, "total": 100}), encoding="utf-8")
    monkeypatch.setattr(workflow, "RAW_NEWS_DIR", tmp_path)
    entries = workflow._read_news_collections("000660", replace(settings, vector_db_path=str(tmp_path)))
    assert entries[0]["next_start"] == 31
    assert entries[0]["returned_count"] == 30
    assert metadata.read_bytes() == before


# 추가 기사·본문이 없으면 즉시 종료
def test_no_progress_stops_early(environment):
    parsed, contexts, collector, search, agent = environment
    collector.return_value = {"new_articles": 0}
    search.return_value = []
    agent.return_value = AgentDecision("collect_news", "없음", parsed.stocks[0], news_start=1)
    _run(environment)
    assert agent.call_count == 1
    assert any("늘지 않아" in note for note in contexts[0]["issues"])


# 반복 검색 표현 차단
def test_repeated_query_is_blocked(environment):
    parsed, contexts, _, search, agent = environment
    search.return_value = _news()
    agent.return_value = AgentDecision("search_news", "표현 변경", parsed.stocks[0], "반도체 실적")
    _run(environment)
    assert search.call_count == 1
    assert any("동일한 검색" in note for note in contexts[0]["issues"])


# 판단 횟수와 인덱싱 호출 상한 각각 강제
def test_decision_and_indexing_limits(environment):
    parsed, contexts, _, search, agent = environment
    search.side_effect = [_news(1), _news(2), _news(3)]
    agent.side_effect = [AgentDecision("search_news", "검색", parsed.stocks[0], f"표현 {i}") for i in range(3)]
    _run(environment)
    assert agent.call_count == 3
    assert any("상한(3회)" in note for note in contexts[0]["issues"])


# 인덱싱 예산 소진 후 두 번째 페이지 실행 차단
def test_indexing_limit(environment):
    parsed, _, collector, search, agent = environment
    collector.return_value, search.return_value = {"new_articles": 1}, _news()
    agent.side_effect = [AgentDecision("collect_news", "첫 수집", parsed.stocks[0], news_start=1),
                         AgentDecision("collect_news", "추가 수집", parsed.stocks[0], news_start=31)]
    _run(environment, replace(settings, agent_max_indexing_calls=1))
    assert collector.call_count == 1


# 최근 동일 조건에서 0건 수집했다면 두 페이지 모두 반복 수집하지 않음
def test_recent_empty_collection_cooldown(environment):
    parsed, contexts, _, _, _ = environment
    contexts[0]["news_collections"] = [{"stock_code": "000660", "start": 1, "article_count": 0,
        "start_date": "2026-11-01", "end_date": "2026-11-13",
        "collected_at": datetime.now(workflow.KST).isoformat()}]
    allowed = workflow._allowed_agent_actions(parsed, contexts, [], 0, settings)
    assert not any(item["action"] == "collect_news" for item in allowed)
    contexts[0]["news_collections"][0]["collected_at"] = (
        datetime.now(workflow.KST) - timedelta(seconds=settings.agent_news_refresh_seconds + 1)).isoformat()
    assert any(item["action"] == "collect_news" for item in workflow._allowed_agent_actions(parsed, contexts, [], 0, settings))


# 기간 내 0건이어도 첫 페이지에 더 최신 기사 30건이 있으면 두 번째 페이지 차단하지 않음
def test_outside_period_page_allows_next_page(environment):
    parsed, contexts, _, _, _ = environment
    contexts[0]["news_collections"] = [{"stock_code": "000660", "start": 1, "article_count": 0,
        "returned_count": 30, "in_period_count": 0, "next_start": 31,
        "start_date": "2026-11-06", "end_date": "2026-11-12",
        "collected_at": datetime.now(workflow.KST).isoformat()}]
    pages = [item["news_start"] for item in workflow._allowed_agent_actions(parsed, contexts, [], 0, settings)
             if item["action"] == "collect_news"]
    assert pages == [31]


# API가 마지막 페이지임을 확인했다면 기간 내 기사가 있어도 추가 페이지를 요청하지 않음
def test_exhausted_page_does_not_offer_more_collection(environment):
    parsed, contexts, _, _, _ = environment
    contexts[0]["news_collections"] = [{"stock_code": "000660", "start": 1, "article_count": 1,
        "returned_count": 1, "in_period_count": 1, "next_start": None,
        "start_date": "2026-11-06", "end_date": "2026-11-12",
        "collected_at": datetime.now(workflow.KST).isoformat()}]
    assert not any(item["action"] == "collect_news" for item in workflow._allowed_agent_actions(
        parsed, contexts, [], 0, settings))


# 첫 페이지에 기간 내 근거가 없어도 다음 페이지가 유효하면 조기 종료 대신 재평가
def test_outside_period_page_does_not_stop_before_second_page(environment):
    parsed, contexts, collector, search, agent = environment
    collector.side_effect = [{"new_articles": 0, "collection": {"next_start": 31}},
                             {"new_articles": 1, "collection": {"next_start": None}}]
    search.side_effect = [[], _news()]
    agent.side_effect = [AgentDecision("collect_news", "목표 기간 탐색", parsed.stocks[0], news_start=1),
                         AgentDecision("collect_news", "다음 페이지", parsed.stocks[0], news_start=31),
                         AgentDecision("respond", "근거 확보")]
    _run(environment)
    assert [call.kwargs["start"] for call in collector.call_args_list] == [1, 31]
    assert contexts[0]["summary"]["news_count"] == 1


# 기간이 다르면 최근 수집 이력으로 잘못 차단하지 않음
def test_collection_period_must_cover_question(environment):
    parsed, contexts, _, _, _ = environment
    contexts[0]["news_collections"] = [{"start": 1, "article_count": 0, "start_date": "2026-11-10",
        "end_date": "2026-11-12", "collected_at": datetime.now(workflow.KST).isoformat()}]
    assert any(item["action"] == "collect_news" for item in workflow._allowed_agent_actions(parsed, contexts, [], 0, settings))


# 주가만 부족할 때 주가만 재조회하며 뉴스 파이프라인은 호출하지 않음
def test_stock_refresh_only(environment, monkeypatch):
    parsed, contexts, collector, search, agent = environment
    refreshed = deepcopy(contexts[0])
    contexts[0]["availability"]["stock"] = False
    contexts[0]["stock"]["current_price"] = None
    build = Mock(return_value=refreshed)
    monkeypatch.setattr(workflow, "_build_analysis_context", build)
    agent.side_effect = [AgentDecision("collect_stock", "주가 부족", parsed.stocks[0]), AgentDecision("respond", "충분")]
    _run(environment)
    assert build.call_count == 1 and build.call_args.kwargs["previous_context"] is not None
    collector.assert_not_called()
    search.assert_not_called()


# 인증 오류는 기록하고 같은 뉴스 수집을 재실행하지 않음
def test_authentication_failure_is_not_retried(environment):
    parsed, contexts, collector, _, agent = environment
    response = httpx.Response(401, request=httpx.Request("GET", "https://example.test"))
    collector.side_effect = httpx.HTTPStatusError("secret-body", request=response.request, response=response)
    agent.return_value = AgentDecision("collect_news", "부족", parsed.stocks[0], news_start=1)
    _run(environment)
    assert collector.call_count == 1
    assert contexts[0]["agent_history"][0]["success"] is False
    assert "HTTP 401" in contexts[0]["issues"][0]
    assert "secret-body" not in str(contexts)


# 전체 시간 초과 시 LLM·수집 시작 전에 종료
def test_deadline_stops_before_tools(environment):
    _run(environment, deadline=time.monotonic() - 1)
    environment[4].assert_not_called()
    environment[2].assert_not_called()
    assert any("TimeoutError" in note for note in environment[1][0]["issues"])


# 보충 후에도 주가가 불충분하더라도 종목당 1회 제한을 넘지 않음
def test_stock_refresh_limit(environment, monkeypatch):
    parsed, contexts, _, _, agent = environment
    refreshed = deepcopy(contexts[0])
    refreshed["availability"]["stock"] = False
    contexts[0]["availability"]["stock"] = False
    contexts[0]["stock"]["current_price"] = None
    build = Mock(return_value=refreshed)
    monkeypatch.setattr(workflow, "_build_analysis_context", build)
    agent.return_value = AgentDecision("collect_stock", "부족", parsed.stocks[0])
    _run(environment)
    assert build.call_count == 1


# 복수 종목 중 부족한 종목에 대한 작업만 실행
def test_only_selected_stock_is_collected(environment):
    parsed, contexts, collector, search, agent = environment
    other = ParsedStock("삼전", "삼성전자", "005930")
    second = deepcopy(contexts[0])
    second["stock"]["stock_code"] = other.stock_code
    second["stock"]["stock_name"] = other.stock_name
    second["news"] = _news(99)
    contexts.append(second)
    parsed = ParsedQuestion("두 종목 비교", (parsed.stocks[0], other), "comparison_reason")
    collector.return_value, search.return_value = {"new_articles": 1}, _news()
    agent.side_effect = [AgentDecision("collect_news", "하이닉스 뉴스 부족", parsed.stocks[0], news_start=1),
                         AgentDecision("respond", "양쪽 확인")]
    workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert collector.call_args.args[0].stock_code == "000660"
    assert collector.call_count == 1

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.app import stock_advice_workflow as workflow
from src.config.settings import settings
from src.domain.advice_models import (AgentDecision, AgentExecutionRecord, ParsedQuestion, ParsedStock,
                                      StockDiscoveryCandidate, StockDiscoveryCriteria, StockDiscoveryEvidence)
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


# 발굴과 후보별 기본 조회를 메모리 대역에 연결하고 실제 외부 요청은 차단
@pytest.fixture
def discovery_environment(environment, monkeypatch):
    parsed = ParsedQuestion("반도체 소재 관련주 3개만 추천해줘", (), "stock_discovery",
                            discovery=StockDiscoveryCriteria("반도체 소재", 3))
    candidate = StockDiscoveryCandidate("SK하이닉스", "소재 사업 원문 근거",
        stock_code="000660", business_verified=True,
        evidence=(StockDiscoveryEvidence("사업 기사", "https://example.test/discovery", "보존할 전체 본문",
                                         excerpt="소재 사업을 영위한다."),))
    discover = Mock(return_value={"candidates": [candidate], "collection": {
        "query": "반도체 소재 관련주", "start": 1, "next_start": None,
    }})

    # 실제 컨텍스트 조립을 사용하되 주가·뉴스 데이터만 모의 제공
    def build(question, stock, config, **kwargs):
        context = build_stock_context(question, {
            "stock_name": stock.stock_name, "stock_code": stock.stock_code, "current_price": 100,
            "base_date": "2026-11-12", "trend": "상승", "source": "toss",
        }, _news(), news_source="hybrid_search")
        context.update(news_period={"start_date": "2026-11-06", "end_date": "2026-11-12"},
            news_collections=[], checked_at=datetime.now(workflow.KST).isoformat(),
            availability={"stock": True, "stock_retryable": True, "news_retryable": True})
        return context

    builder = Mock(side_effect=build)
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    monkeypatch.setattr(workflow, "discover_stocks", discover)
    monkeypatch.setattr(workflow, "_build_analysis_context", builder)
    monkeypatch.setattr(workflow, "collect_stock_price", Mock(side_effect=AssertionError("No real API")))
    agent = environment[4]
    agent.side_effect = [AgentDecision("discover_stocks", "테마 후보 필요", news_start=1), AgentDecision("respond", "근거 확보")]
    return parsed, candidate, discover, builder, agent


# 발굴 전에는 코드 없는 발굴·종료만 허용하고 실패 이력도 재발굴을 차단
@pytest.mark.parametrize("success", [True, False])
def test_discovery_allowed_once(discovery_environment, success):
    parsed = discovery_environment[0]
    config = replace(settings, agent_max_discoveries=5)
    assert workflow._allowed_agent_actions(parsed, [], [], 0, config) == [
        {"action": "discover_stocks", "news_start": 1}, {"action": "stop"}]
    history = [AgentExecutionRecord("discover_stocks", None, success, "시도")]
    assert workflow._allowed_agent_actions(parsed, [], history, 0, config) == [{"action": "stop"}]


# 발굴 후보의 출처·선정 이유·전체 본문과 실행 이력을 보존하고 기존 분석에 전달
def test_discovery_connects_workflow(discovery_environment, monkeypatch):
    parsed, candidate, discover, builder, agent = discovery_environment
    generator = Mock(wraps=workflow.generate_stock_advice)
    monkeypatch.setattr(workflow, "generate_stock_advice", generator)
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    discover.assert_called_once()
    assert discover.call_args.args == (parsed.discovery,)
    builder.assert_called_once()
    executed_question, stock = builder.call_args.args[:2]
    assert executed_question.stocks == ()
    assert executed_question.discovered_candidates == (candidate,)
    assert stock.role == "discovered"
    context = generator.call_args.args[0]
    assert context["question"]["stocks"] == []
    assert context["discovery_candidate"]["evidence"][0]["content"] == "보존할 전체 본문"
    assert [record["action"] for record in context["agent_history"]] == [
        "discover_stocks", "respond"]
    assert "최종 추천 선정 전 단계" in answer
    assert "소재 사업 원문 근거" in answer and "https://example.test/discovery" in answer
    assert agent.call_count == 2
    assert agent.call_args_list[0].args[0].stocks == ()
    assert discover.call_args.kwargs["deadline"] == builder.call_args.kwargs["deadline"]


# 빈 결과는 발굴 성공 이력과 함께 즉시 종료하고 종목별 조회를 시작하지 않음
def test_empty_discovery_stops(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    discover.return_value = {"candidates": [], "collection": {"start": 1, "next_start": None}}
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert "검증된 분석 후보를 확보하지 못했습니다" in answer
    assert "발굴 재시도는 하지 않습니다" in answer
    discover.assert_called_once()
    builder.assert_not_called()
    assert agent.call_count == 1


# 발굴 오류·시간 초과 시 원문 에러를 노출하거나 발굴을 반복하지 않음
@pytest.mark.parametrize("error", [ValueError("private-body"), TimeoutError("private-body")])
def test_failed_discovery_stops(discovery_environment, error):
    parsed, _, discover, builder, agent = discovery_environment
    discover.side_effect = error
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert "종목 발굴 실패" in answer and "private-body" not in answer
    discover.assert_called_once()
    builder.assert_not_called()
    assert agent.call_count == 1


# Agent가 두 번째 발굴을 선택해도 서버의 허용 목록 검증에서 차단
def test_duplicate_discovery_is_blocked(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    agent.side_effect = None
    agent.return_value = AgentDecision("discover_stocks", "테마 후보 필요", news_start=1)
    contexts = []
    workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    discover.assert_called_once()
    builder.assert_called_once()
    assert any("Agent 처리 실패" in issue for issue in contexts[0]["issues"])


# 발굴도 전체 Agent 결정 횟수에 포함하고 후속 판단에 새 예산을 부여하지 않음
def test_discovery_shares_decision_budget(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    contexts = []
    workflow._run_agent_loop(parsed, contexts, replace(settings, agent_max_decisions=1), time.monotonic() + 60)
    assert agent.call_count == 1
    discover.assert_called_once()
    builder.assert_called_once()
    assert "상한(1회)" in " ".join(contexts[0]["issues"])


# 시간 예산이 이미 소진되면 판단·발굴·주가 조회 모두 시작하지 않음
def test_discovery_deadline_blocks_all_tools(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    _, notes = workflow._run_agent_loop(parsed, [], settings, time.monotonic() - 1)
    assert "TimeoutError" in " ".join(notes)
    agent.assert_not_called()
    discover.assert_not_called()
    builder.assert_not_called()


# 검증되지 않은 후보나 검토 상한 초과 결과는 주가 조회에 전달하지 않음
@pytest.mark.parametrize("invalid", ["unverified", "missing_code", "missing_evidence", "excessive"])
def test_discovery_validates_execution_targets(discovery_environment, invalid):
    parsed, candidate, discover, builder, _ = discovery_environment
    changes = {"unverified": {"business_verified": False}, "missing_code": {"stock_code": None},
               "missing_evidence": {"evidence": ()}, "excessive": {}}
    discover.return_value["candidates"] = [replace(candidate, **changes[invalid])] * (11 if invalid == "excessive" else 1)
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert "종목 발굴 실패" in answer
    builder.assert_not_called()


# 후보별 조회 실패를 기록하되 다른 검증 후보의 분석은 계속 진행
def test_discovered_stock_failure_preserves_provenance(discovery_environment):
    parsed, candidate, _, builder, agent = discovery_environment
    builder.side_effect = httpx.ConnectError("private-body")
    agent.side_effect = [AgentDecision("discover_stocks", "후보 필요", news_start=1), AgentDecision("stop", "주가 조회 실패")]
    contexts = []
    workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert contexts[0]["discovery_candidate"]["stock_code"] == candidate.stock_code
    assert contexts[0]["availability"]["stock"] is False
    assert "private-body" not in str(contexts)


# 첫 후보 조회에 실패해도 두 번째 검증 후보를 누락하지 않고 요청 개수로 미리 잘라내지 않음
def test_discovery_analyzes_other_candidates_after_failure(discovery_environment):
    parsed, candidate, discover, builder, agent = discovery_environment
    parsed = replace(parsed, discovery=StockDiscoveryCriteria("반도체 소재", 1))
    other = replace(candidate, stock_name="삼성전자", stock_code="005930")
    discover.return_value["candidates"] = [candidate, other]
    original_build = builder.side_effect

    # 첫 종목만 실패시키고 나머지는 기존 모의 컨텍스트로 조립
    def build(question, stock, config, **kwargs):
        if stock.stock_code == "000660":
            raise httpx.ConnectError("offline")
        return original_build(question, stock, config, **kwargs)

    builder.side_effect = build
    agent.side_effect = [AgentDecision("discover_stocks", "후보 필요", news_start=1), AgentDecision("stop", "부분 근거만 확보")]
    contexts = []
    result, _ = workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert result.stocks == ()
    assert len(result.discovered_candidates) == len(contexts) == builder.call_count == 2
    assert contexts[0]["availability"]["stock"] is False
    assert contexts[1]["availability"]["stock"] is True
    assert contexts[1]["stock"]["stock_code"] == "005930"


# 발굴 전 질문에 종목이 있더라도 검증된 후보로 취급해 선조회하지 않음
def test_discovery_does_not_use_question_stocks(discovery_environment, monkeypatch):
    parsed, _, discover, builder, agent = discovery_environment
    parsed = replace(parsed, stocks=(ParsedStock("삼전", "삼성전자", "005930"),))
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    discover.return_value = {"candidates": [], "collection": {"start": 1, "next_start": None}}
    workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    builder.assert_not_called()
    assert agent.call_count == 1


# 샘플 제공자는 기존 지정 종목 분석만 유지하고 실제 발굴·가짜 후보 대체는 하지 않음
def test_sample_discovery_is_explicitly_unsupported(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="sample"))
    assert "STOCK_API_PROVIDER=toss" in answer
    discover.assert_not_called()
    builder.assert_not_called()
    agent.assert_not_called()


# 첫 페이지에 원문 근거가 없어도 다음 페이지가 있으면 발굴 1회 안에서 두 번째 페이지 조회
def test_discovery_second_page_recovers_candidates(discovery_environment, monkeypatch):
    parsed, candidate, discover, builder, agent = discovery_environment
    discover.side_effect = [
        {"candidates": [], "collection": {"start": 1, "next_start": 31, "query": "반도체 소재 관련주"}},
        {"candidates": [candidate], "collection": {"start": 31, "next_start": None, "query": "반도체 소재 관련주"}},
    ]
    agent.side_effect = [AgentDecision("discover_stocks", "첫 페이지", news_start=1),
                        AgentDecision("discover_stocks", "추가 페이지", news_start=31), AgentDecision("respond", "충분")]
    generator = Mock(wraps=workflow.generate_stock_advice)
    monkeypatch.setattr(workflow, "generate_stock_advice", generator)
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert [call.kwargs["start"] for call in discover.call_args_list] == [1, 31]
    assert discover.call_args_list[0].kwargs["deadline"] == discover.call_args_list[1].kwargs["deadline"]
    history = generator.call_args.args[0]["agent_history"]
    assert [record["news_start"] for record in history if record["action"] == "discover_stocks"] == [1, 31]
    assert all(record["success"] for record in history)
    assert "SK하이닉스" in answer
    builder.assert_called_once()
    assert agent.call_count == 3


# 두 페이지 모두 비어 있어도 세 번째 페이지나 새로운 발굴을 실행하지 않음
def test_discovery_empty_pages_stop_at_two(discovery_environment):
    parsed, _, discover, builder, agent = discovery_environment
    discover.side_effect = [
        {"candidates": [], "collection": {"start": 1, "next_start": 31}},
        {"candidates": [], "collection": {"start": 31, "next_start": 31}},
    ]
    agent.side_effect = [AgentDecision("discover_stocks", "첫 페이지", news_start=1),
                        AgentDecision("discover_stocks", "추가 페이지", news_start=31)]
    answer = workflow.run_stock_advice_workflow(parsed.original_question, replace(settings, stock_api_provider="toss"))
    assert "검증된 분석 후보를 확보하지 못했습니다" in answer
    assert discover.call_count == agent.call_count == 2
    builder.assert_not_called()


# 요청 수 충족·마지막 페이지·페이지 예산 소진·검토 후보 상한 도달 시 추가 조회 생략
@pytest.mark.parametrize("requested_count,next_start,pages,limit", [
    (1, 31, 2, 10), (3, None, 2, 10), (3, 31, 1, 10), (3, 31, 2, 1),
])
def test_discovery_second_page_is_conditional(discovery_environment, requested_count, next_start, pages, limit):
    parsed, _, discover, _, _ = discovery_environment
    parsed = replace(parsed, discovery=StockDiscoveryCriteria("반도체 소재", requested_count))
    discover.return_value["collection"]["next_start"] = next_start
    config = replace(settings, agent_max_news_pages=pages, agent_max_discovery_candidates=limit)
    workflow._run_agent_loop(parsed, [], config, time.monotonic() + 60)
    discover.assert_called_once()


# 페이지 간 동일 종목과 URL 근거는 병합하고 새로운 검증 후보만 추가
def test_discovery_merges_page_candidates(discovery_environment):
    parsed, candidate, discover, builder, agent = discovery_environment
    extra_evidence = replace(candidate.evidence[0], url="https://example.test/second")
    duplicate = replace(candidate, evidence=candidate.evidence + (extra_evidence,))
    other = replace(candidate, stock_name="삼성전자", stock_code="005930")
    discover.side_effect = [
        {"candidates": [candidate], "collection": {"start": 1, "next_start": 31}},
        {"candidates": [duplicate, other], "collection": {"start": 31, "next_start": None}},
    ]
    agent.side_effect = [AgentDecision("discover_stocks", "첫 페이지", news_start=1),
                        AgentDecision("discover_stocks", "추가 페이지", news_start=31), AgentDecision("respond", "충분")]
    contexts = []
    result, _ = workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert len(result.discovered_candidates) == builder.call_count == len(contexts) == 2
    assert [item.stock_code for item in result.discovered_candidates] == ["000660", "005930"]
    assert len(result.discovered_candidates[0].evidence) == 2
    assert len({item.url for item in result.discovered_candidates[0].evidence}) == 2


# 두 번째 페이지 실패 시 먼저 확보한 후보를 버리지 않고 실패 이력·한계를 전달
def test_discovery_second_page_failure_keeps_first_candidates(discovery_environment):
    parsed, candidate, discover, builder, agent = discovery_environment
    discover.side_effect = [
        {"candidates": [candidate], "collection": {"start": 1, "next_start": 31}},
        httpx.ConnectError("private-response"),
    ]
    agent.side_effect = [AgentDecision("discover_stocks", "첫 페이지", news_start=1),
                        AgentDecision("discover_stocks", "추가 페이지", news_start=31)]
    contexts = []
    result, notes = workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert result.discovered_candidates == (candidate,)
    assert "종목 발굴 실패" in " ".join(notes)
    assert "private-response" not in str(contexts)
    page_history = [record for record in contexts[0]["agent_history"] if record["action"] == "discover_stocks"]
    assert [record["success"] for record in page_history] == [True, False]
    builder.assert_called_once()


# 두 번째 페이지 직전 예산 소진 시 외부 요청 없이 기존 후보와 중단 사유 유지
def test_discovery_second_page_timeout_keeps_first_candidates(discovery_environment, monkeypatch):
    parsed, candidate, discover, _, _ = discovery_environment
    discover.return_value["collection"]["next_start"] = 31
    monkeypatch.setattr(workflow, "remaining_timeout", Mock(side_effect=[20, 20, 20, TimeoutError("expired"), 20]))
    contexts = []
    result, notes = workflow._run_agent_loop(parsed, contexts, settings, time.monotonic() + 60)
    assert result.discovered_candidates == (candidate,)
    assert "TimeoutError" in " ".join(notes)
    discover.assert_called_once()
    assert contexts[0]["agent_history"][0]["news_start"] == 1


# 실제 수집·후보 검증 코드를 연결하여 미지원 언론사 30건 이후 두 번째 페이지에서 후보 확보
def test_discovery_pagination_with_mocked_http(discovery_environment, monkeypatch):
    from src.collectors import news_collector
    from src.services.stock_discovery_service import discover_stocks

    parsed, _, _, builder, agent = discovery_environment
    agent.side_effect = [AgentDecision("discover_stocks", "첫 페이지", news_start=1),
                        AgentDecision("discover_stocks", "추가 페이지", news_start=31), AgentDecision("respond", "충분")]
    quote = "SK하이닉스는 반도체 소재 관련 사업을 영위한다."
    monkeypatch.setattr(news_collector, "extract", Mock(return_value=quote))
    llm = Mock()
    llm.with_options.return_value = llm
    llm.responses.create.return_value = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="function_call", name="submit_discovery_candidates", arguments=json.dumps({"candidates": [{
            "stock_name": "SK하이닉스", "selection_reason": "원문 내 사업 근거",
            "evidence": [{"source_index": 0, "excerpt": quote, "relation": "direct"}],
        }]}))])
    requested_pages = []

    # 네이버 검색 두 페이지와 허용 언론사의 robots·원문 요청만 모의 처리
    def respond(request):
        if request.url.host == "openapi.naver.com":
            start = int(request.url.params["start"])
            requested_pages.append(start)
            assert request.url.params["query"] == "반도체 소재 관련주"
            items = [{"title": "소재 사업 기사", "pubDate": "Thu, 05 Nov 2026 12:00:00 +0900",
                "originallink": ("https://not-allowed.example/1" if start == 1
                                 else "https://www.hankyung.com/article/1")} for _ in range(30 if start == 1 else 1)]
            return httpx.Response(200, json={"total": 60, "items": items})
        assert request.url.host == "www.hankyung.com"
        text = "User-agent: *\nDisallow:\n" if request.url.path == "/robots.txt" else "<article>본문</article>"
        return httpx.Response(200, text=text)

    config = replace(settings, stock_api_provider="toss", news_api_provider="naver",
                     news_api_key="test", news_api_secret="test", openai_api_key=None)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        # workflow가 전달한 시작 위치·예산을 실제 발굴 함수에 그대로 전달
        def discover(criteria, **kwargs):
            return discover_stocks(criteria, client=llm, http_client=client,
                stock_infos=[{"stock_name": "SK하이닉스", "stock_code": "000660"}], **kwargs)

        monkeypatch.setattr(workflow, "discover_stocks", discover)
        answer = workflow.run_stock_advice_workflow(parsed.original_question, config)
    assert requested_pages == [1, 31]
    llm.responses.create.assert_called_once()
    builder.assert_called_once()
    assert "원문 내 사업 근거" in answer

from dataclasses import replace
from datetime import datetime
import time
from unittest.mock import Mock

import httpx
import pytest

from src.app import stock_advice_workflow as workflow
from src.config.settings import settings
from src.domain.advice_models import (AgentDecision, AgentExecutionRecord, ParsedQuestion, StockDiscoveryCandidate,
                                      StockDiscoveryCriteria, StockDiscoveryEvidence)
from src.services.stock_context_service import build_stock_context


# 발굴·주가·임베딩을 모두 대체하고 실제 workflow·컨텍스트 조립만 실행
@pytest.fixture
def environment(monkeypatch):
    parsed = ParsedQuestion("소재 관련주 3개 추천", (), "stock_discovery", discovery=StockDiscoveryCriteria("소재", 3))
    candidates = [StockDiscoveryCandidate(f"소재기업{index}", "사업 원문 근거", stock_code=f"{index:06}",
        business_verified=True, evidence=(StockDiscoveryEvidence("사업 기사", f"https://example.test/theme/{index}",
            "보존할 전체 본문", publisher="경제신문", published_date="2026-11-17", excerpt="직접 사업 근거"),))
        for index in range(1, 4)]
    config = replace(settings, stock_api_provider="toss", openai_api_key=None)
    collections = {}

    # 실제 수집 이력과 동일 필드를 갖는 모의 collection 구성
    def collection(stock=None, start=1, next_start=None):
        return {"query": stock.stock_name if stock else "소재 관련주",
            "stock_name": stock.stock_name if stock else None, "stock_code": stock.stock_code if stock else None,
            "start_date": "2026-11-11", "end_date": "2026-11-17", "collected_at": datetime.now(workflow.KST).isoformat(),
            "start": start, "display": 30, "returned_count": 30, "in_period_count": 30, "next_start": next_start}

    # 검증된 종목의 가격과 기존 검색 결과 0건을 모의 제공
    def build(question, stock, app_settings, **kwargs):
        context = build_stock_context(question, {"stock_name": stock.stock_name, "stock_code": stock.stock_code,
            "current_price": 100, "base_date": "2026-11-17", "trend": "상승", "source": "toss"}, [],
            news_source="hybrid_search")
        context.update(news_period={"start_date": "2026-11-11", "end_date": "2026-11-17"}, news_collections=[],
            checked_at=datetime.now(workflow.KST).isoformat(), news_search_status="empty", news_search_error=None,
            availability={"stock": True, "stock_retryable": True, "news_retryable": True})
        return context

    # 종목별 저장 함수의 결과와 수집 이력을 모의 제공
    def collect(stock, **kwargs):
        info = collection(stock, kwargs["start"])
        info["article_count"] = 1
        collections.setdefault(stock.stock_code, []).append(info)
        return {"collection": info, "new_articles": 1}

    # 발굴과 후보별 수집·응답을 각각의 실행 이력에 따라 선택
    def decide(question, contexts, history, allowed, **kwargs):
        if not contexts:
            page = next(item["news_start"] for item in allowed if item["action"] == "discover_stocks")
            return AgentDecision("discover_stocks", "후보 발굴", news_start=page)
        if not contexts[0]["news"]:
            return AgentDecision("collect_news", "뉴스 보충", question.analysis_stocks[0], news_start=1)
        return AgentDecision("respond", "근거 확보")

    discover = Mock(return_value={"candidates": candidates, "collection": collection()})
    builder, collector, agent = Mock(side_effect=build), Mock(side_effect=collect), Mock(side_effect=decide)
    search = Mock(side_effect=lambda query, **kwargs: [{"title": "종목별 근거", "content": "근거 본문",
        "url": f"https://example.test/stock/{kwargs['stock_code']}", "published_date": "2026-11-17"}])
    generator = Mock(wraps=workflow.generate_stock_comparison_advice)
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    monkeypatch.setattr(workflow, "discover_theme_stocks", discover)
    monkeypatch.setattr(workflow, "_build_analysis_context", builder)
    monkeypatch.setattr(workflow, "collect_and_index_news", collector)
    monkeypatch.setattr(workflow, "decide_next_action", agent)
    monkeypatch.setattr(workflow, "search_news_hybrid", search)
    monkeypatch.setattr(workflow, "_missing_news_index", Mock(return_value=False))
    monkeypatch.setattr(workflow, "_read_news_collections", lambda code, config: list(collections.get(code, [])))
    monkeypatch.setattr(workflow, "generate_stock_comparison_advice", generator)
    return dict(parsed=parsed, candidates=candidates, config=config, collection=collection, discover=discover,
                builder=builder, collector=collector, agent=agent, search=search, generator=generator, collections=collections)


# 세 후보 모두 기존 분석 함수·저장 파이프라인을 공유하고 예산·원래 질문·발굴 근거를 구분
def test_all_candidates_use_shared_analysis(environment):
    env = environment
    answer = workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert env["discover"].call_count == 1
    assert env["collector"].call_count == env["builder"].call_count == env["search"].call_count == 3
    assert env["agent"].call_count == 7  # 발굴 1 + 각 후보 수집·응답 2
    contexts = env["generator"].call_args.args[0]
    assert len(contexts) == 3
    for context in contexts:
        assert context["question"]["stocks"] == []
        assert context["discovery_candidate"]["evidence"][0]["content"] == "보존할 전체 본문"
        assert [record["action"] for record in context["agent_history"]] == ["collect_news", "respond"]
        assert [record["action"] for record in context["discovery_history"]] == ["discover_stocks"]
        record = context["agent_history"][0]
        assert record["collection"]["query"] == context["stock"]["stock_name"]
        assert "관련주" not in record["collection"]["query"]
        assert context["summary"]["news_count"] == 1
    assert "최종 추천 종목 선정 전 단계" in answer
    assert "발굴 이유: 사업 원문 근거" in answer
    assert env["parsed"].discovered_candidates == ()


# 근거가 늘지 않은 정상 종료는 오류 사유가 아닌 처리 안내로 전달
def test_unchanged_evidence_is_reported_as_processing_notice(environment):
    env = environment
    parsed = replace(env["parsed"], discovered_candidates=(env["candidates"][0],))
    stock = parsed.analysis_stocks[0]
    context = env["builder"](parsed, stock, env["config"])
    context["news"] = env["search"]("질문", stock_code=stock.stock_code)
    context["news_search_status"] = "found"
    env["collector"].side_effect = None
    env["collector"].return_value = {"new_articles": 0, "collection": env["collection"](stock)}
    env["agent"].side_effect = None
    env["agent"].return_value = AgentDecision("collect_news", "근거 보충", stock, news_start=1)

    result = workflow._run_agent_loop(parsed, [context], env["config"], time.monotonic() + 60)

    notice = "추가 기사·근거가 늘지 않아 반복 조사를 종료했습니다."
    assert context["processing_notices"] == [notice]
    assert context["issues"] == [] and result["notes"] == []
    answer = workflow.generate_stock_advice(context)
    assert f"처리 안내:\n- {notice}" in answer
    assert "데이터 확인 한계:" not in answer
    assert env["agent"].call_count == 1


# 첫 페이지 원문 0건이어도 다음 반복의 Agent 결정으로 두 번째 페이지 조회
def test_discovery_pages_share_agent_loop(environment):
    env = environment
    env["discover"].side_effect = [
        {"candidates": [], "collection": env["collection"](start=1, next_start=31)},
        {"candidates": env["candidates"], "collection": env["collection"](start=31)},
    ]
    workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert [call.kwargs["start"] for call in env["discover"].call_args_list] == [1, 31]
    assert env["collector"].call_count == 3
    history = env["generator"].call_args.args[0][0]["discovery_history"]
    assert [record["news_start"] for record in history] == [1, 31]
    assert all(set(record["collection"]) == set(history[0]["collection"]) for record in history)


# 각 후보가 2페이지·3회 결정을 사용해도 다른 후보의 결정·인덱싱 기회를 소진하지 않음
def test_each_candidate_has_independent_budgets(environment, monkeypatch):
    env = environment
    loop = Mock(wraps=workflow._run_agent_loop)
    monkeypatch.setattr(workflow, "_run_agent_loop", loop)

    # 첫 페이지 0건은 다음 페이지 가능 이력으로 남김
    def collect(stock, **kwargs):
        first = kwargs["start"] == 1
        info = env["collection"](stock, kwargs["start"], 31 if first else None)
        info["article_count"] = 0 if first else 1
        env["collections"].setdefault(stock.stock_code, []).append(info)
        return {"collection": info, "new_articles": 0 if first else 1}

    # 서버가 허용한 종목별 다음 페이지를 선택
    def decide(question, contexts, history, allowed, **kwargs):
        if not contexts:
            return AgentDecision("discover_stocks", "발굴", news_start=1)
        if contexts[0]["news"]:
            return AgentDecision("respond", "충분")
        page = next(item["news_start"] for item in allowed if item["action"] == "collect_news")
        return AgentDecision("collect_news", "추가 페이지", question.analysis_stocks[0], news_start=page)

    # 실제 페이지별 결과에 맞춰 첫 페이지 검색 결과는 0건
    def search(query, **kwargs):
        page = env["collections"][kwargs["stock_code"]][-1]["start"]
        return [] if page == 1 else [{"title": "근거", "content": "본문", "url": "https://example.test/news",
                                      "published_date": "2026-11-17"}]

    env["collector"].side_effect, env["agent"].side_effect, env["search"].side_effect = collect, decide, search
    workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert env["agent"].call_count == 10
    assert [call.kwargs["start"] for call in env["collector"].call_args_list] == [1, 31] * 3
    assert loop.call_count == 4  # 발굴 1회 실행 + 후보별 분석 루프 3회 실행
    assert all(len(call.args[1]) == 1 for call in loop.call_args_list[1:])
    contexts = env["generator"].call_args.args[0]
    assert all(len(context["agent_history"]) == 3 for context in contexts)


# 이미 검색 근거가 충분한 후보는 추가 수집·저장을 생략
def test_existing_evidence_skips_new_collection(environment):
    env = environment
    original = env["builder"].side_effect

    # 기존 인덱스의 기사 근거를 갖춘 컨텍스트 제공
    def build(*args, **kwargs):
        context = original(*args, **kwargs)
        context["news"] = [{"title": "기존 기사", "content": "본문", "published_date": "2026-11-17"}]
        context["summary"] = {"news_count": 1, "news_chunk_count": 1}
        return context

    env["builder"].side_effect = build
    workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    env["collector"].assert_not_called()
    assert env["agent"].call_count == 4


# 페이지 중복·마지막 페이지·후보 수 충족·실패 이력·페이지 예산을 서버가 직접 제한
@pytest.mark.parametrize("success,next_page,page_count,candidate_count,expected", [
    (True, 31, 2, 0, 31), (False, 31, 2, 0, None), (True, None, 2, 0, None),
    (True, 31, 1, 0, None), (True, 31, 2, 3, None),
])
def test_discovery_allowed_pages(environment, success, next_page, page_count, candidate_count, expected):
    env = environment
    parsed = replace(env["parsed"], discovered_candidates=tuple(env["candidates"][:candidate_count]))
    history = [AgentExecutionRecord("discover_stocks", None, success, "조회", news_start=1,
                                   collection=env["collection"](next_start=next_page))]
    allowed = workflow._allowed_agent_actions(parsed, [], history, 0,
                                               replace(env["config"], agent_max_news_pages=page_count))
    assert [item["news_start"] for item in allowed if item["action"] == "discover_stocks"] == (
        [] if expected is None else [expected])
    history.append(AgentExecutionRecord("discover_stocks", None, True, "추가", news_start=31))
    assert workflow._allowed_agent_actions(parsed, [], history, 0, env["config"]) == [{"action": "stop"}]


# 발굴 빈 결과·예외는 종목별 조회로 넘기지 않고 안전한 이유와 함께 종료
@pytest.mark.parametrize("failure", [False, True])
def test_empty_or_failed_discovery(environment, failure):
    env = environment
    if failure:
        env["discover"].side_effect = ValueError("private-response")
    else:
        env["discover"].return_value["candidates"] = []
    answer = workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert "검증된 분석 후보를 확보하지 못했습니다" in answer
    assert "private-response" not in answer
    env["builder"].assert_not_called()
    env["collector"].assert_not_called()


# 추가 페이지 실패 시 이전 검증 후보를 유지하고 후보별 분석은 새 예산으로 진행
def test_partial_discovery_is_preserved(environment):
    env = environment
    env["discover"].side_effect = [
        {"candidates": env["candidates"][:1], "collection": env["collection"](next_start=31)},
        httpx.ConnectError("private-response"),
    ]
    answer = workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert "소재기업1" in answer and "종목 발굴 실패" in answer
    assert "private-response" not in answer
    assert env["builder"].call_count == env["collector"].call_count == 1


# 중단 이유는 실제 검색 실패로 고정하여 과거 수집 0건이라는 LLM 설명을 사실처럼 노출하지 않음
def test_failed_search_is_not_reported_as_no_articles(environment):
    env = environment
    original = env["builder"].side_effect

    # 인덱스 오류로 검색하지 못한 후보 컨텍스트 구성
    def build(*args, **kwargs):
        context = original(*args, **kwargs)
        context.update(news_search_status="failed", news_search_error="뉴스 검색 실패 (ValueError).")
        context["availability"]["news_retryable"] = False
        context["issues"] = ["뉴스 검색 실패 (ValueError)."]
        return context

    # 발굴 후에는 서버가 허용한 종료 작업만 선택
    def decide(question, contexts, history, allowed, **kwargs):
        return (AgentDecision("stop", "이미 기사 0건이라 수집 불가") if contexts else
                AgentDecision("discover_stocks", "발굴", news_start=1))

    env["builder"].side_effect, env["agent"].side_effect = build, decide
    answer = workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert "기사 0건이라 수집 불가" not in answer
    assert "뉴스 검색 실패로 기사 유무를 판단할 수 없습니다" in answer
    assert "질문 기간에 맞는 뉴스 근거가 부족합니다" not in answer
    env["collector"].assert_not_called()


# 발굴 단계 시간이 소진되면 Agent·외부 수집을 시작하지 않음
def test_expired_discovery_budget(environment):
    env = environment
    result = workflow._run_agent_loop(env["parsed"], [], env["config"], time.monotonic() - 1)
    assert "TimeoutError" in " ".join(result["notes"])
    env["agent"].assert_not_called()
    env["discover"].assert_not_called()


# 첫 후보 주가 조회가 실패해도 다른 후보의 근거 수집·분석을 계속 수행
def test_one_candidate_failure_does_not_block_others(environment):
    env = environment
    original = env["builder"].side_effect

    # 첫 종목만 가격 조회 오류를 발생시키고 다른 후보는 정상 컨텍스트 반환
    def build(question, stock, config, **kwargs):
        if stock.stock_code == "000001":
            raise httpx.ConnectError("private-price-response")
        return original(question, stock, config, **kwargs)

    original_decide = env["agent"].side_effect

    # 가격이 없는 후보는 종료하고 다른 후보의 판단은 기존 규칙 사용
    def decide(question, contexts, history, allowed, **kwargs):
        if contexts and not contexts[0]["availability"]["stock"]:
            return AgentDecision("stop", "가격 미확보")
        return original_decide(question, contexts, history, allowed, **kwargs)

    env["builder"].side_effect, env["agent"].side_effect = build, decide
    answer = workflow.run_stock_advice_workflow(env["parsed"].original_question, env["config"])
    assert env["builder"].call_count == 3
    assert env["collector"].call_count == 2
    assert {call.args[0].stock_code for call in env["collector"].call_args_list} == {"000002", "000003"}
    contexts = env["generator"].call_args.args[0]
    assert contexts[0]["discovery_candidate"]["stock_code"] == "000001"
    assert contexts[0]["availability"]["stock"] is False
    assert "private-price-response" not in answer

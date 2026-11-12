from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock
import time

import httpx
from openai import APIConnectionError
import pytest

from src.app.rag_indexing_workflow import collect_and_index_news
from src.config.settings import settings
from src.domain.advice_models import ParsedStock
from src.indexing.news_indexer import load_news_indexes


# 수집·전처리·청킹·실제 FAISS/BM25를 임시 경로에 실행하고 네트워크·임베딩만 대역 제공
@pytest.fixture
def environment(tmp_path):
    client = Mock()
    client.with_options.return_value = client
    client.embeddings.create.side_effect = lambda **options: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[1, i + 1]) for i in range(len(options["input"]))
    ])
    kiwi = Mock()
    kiwi.tokenize.side_effect = lambda text: [SimpleNamespace(form=word, tag="NNG") for word in text.split()]
    config = replace(settings, openai_api_key=None, stock_api_key=None, stock_api_secret=None,
                     news_api_provider="naver", news_api_key="test", news_api_secret="test",
                     vector_db_provider="faiss", openai_embedding_model="test-model")
    kwargs = dict(app_settings=config, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
                  chunks_dir=tmp_path / "chunks", faiss_dir=tmp_path / "faiss",
                  bm25_path=tmp_path / "tokens.json", client=client, kiwi=kiwi)
    return kwargs


# 언론사 원문 조회 없는 API 요약 기사 응답 구성
def _payload(number=1):
    return {"items": [{"title": "반도체 매출 증가" if number == 1 else "자동차 공장 중단",
                       "description": "반도체 매출 10조원" if number == 1 else "자동차 생산 3만대 중단",
                       "originallink": f"https://example.test/{number}", "link": f"https://example.test/{number}",
                       "pubDate": "Thu, 02 Jan 2020 10:00:00 +0900"}]}


# 해당 기간의 뉴스 수집·인덱싱 파이프라인 실행
def _run(kwargs, payload, start=1):
    # 페이지 요청과 인증 헤더는 외부로 전송하지 않는 MockTransport 사용
    def respond(request):
        assert request.url.params["start"] == str(start)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as connection:
        return collect_and_index_news(ParsedStock("하이닉스", "SK하이닉스", "000660"),
                                      "2020-01-01", "2020-01-03", start=start,
                                      http_client=connection, deadline=time.monotonic() + 30, **kwargs)


# 두 페이지의 수집 시각·기간이 인덱스 메타데이터에 남고 재로딩 가능
def test_pipeline_pages_preserve_collection_history(environment):
    first = _run(environment, _payload(1))
    second = _run(environment, _payload(2), start=31)
    assert first["new_articles"] == second["new_articles"] == 1
    faiss, bm25, metadata = load_news_indexes(environment["faiss_dir"], environment["bm25_path"])
    assert faiss.ntotal == bm25.corpus_size == 2
    assert {item["start"] for item in metadata["collections"]} == {1, 31}
    assert all(item["collected_at"] and item["stock_code"] == "000660" for item in metadata["collections"])
    assert all(item["article_count"] == 1 for item in metadata["collections"])


# 최초 수집 결과가 0건이어도 이력은 남고 유료 임베딩·가짜 빈 FAISS 생성 없음
def test_empty_collection_is_persisted_without_embedding(environment):
    result = _run(environment, {"items": []})
    assert result["new_articles"] == 0 and result["chunk_count"] == 0
    environment["client"].embeddings.create.assert_not_called()
    metadata = json.loads((environment["faiss_dir"] / "stock_advisor_metadata.json").read_text(encoding="utf-8"))
    assert metadata["collections"][0]["article_count"] == 0
    assert not (environment["faiss_dir"] / "stock_advisor.index").exists()
    assert not environment["bm25_path"].exists()


# 기존 정상 인덱스가 있으면 빈 추가 수집은 기존 파일을 유지하고 이력만 추가
def test_empty_page_updates_history_without_reembedding(environment):
    _run(environment, _payload())
    before = (environment["faiss_dir"] / "stock_advisor.index").read_bytes()
    result = _run(environment, {"items": []}, start=31)
    assert result["rebuilt"] is False
    assert environment["client"].embeddings.create.call_count == 1
    assert (environment["faiss_dir"] / "stock_advisor.index").read_bytes() == before
    _, _, metadata = load_news_indexes(environment["faiss_dir"], environment["bm25_path"])
    assert len(metadata["collections"]) == 2


# 청크·문자 예산을 초과하면 유료 요청 전에 거부
@pytest.mark.parametrize("limit", ["chunks", "characters"])
def test_indexing_budget_stops_before_paid_request(environment, limit):
    if limit == "chunks":
        environment["app_settings"] = replace(environment["app_settings"], agent_max_index_chunks=1)
        payload = {"items": _payload(1)["items"] + _payload(2)["items"]}
    else:
        environment["app_settings"] = replace(environment["app_settings"], agent_max_index_characters=1)
        payload = _payload()
    with pytest.raises(ValueError, match="limit exceeded"):
        _run(environment, payload)
    environment["client"].embeddings.create.assert_not_called()


# 임베딩 실패 시 새로운 수집을 성공 이력으로 기록하지 않고 기존 인덱스 유지
def test_failed_indexing_keeps_previous_freshness(environment):
    _run(environment, _payload(1))
    path = environment["faiss_dir"] / "stock_advisor_metadata.json"
    before = path.read_bytes()
    environment["client"].embeddings.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://example.test"))
    with pytest.raises(ValueError, match="APIConnectionError"):
        _run(environment, _payload(2), start=31)
    assert path.read_bytes() == before


# 전체 처리 시간이 끝났으면 수집·API 요청 전에 차단
def test_expired_deadline_stops_pipeline(environment):
    with pytest.raises(TimeoutError):
        collect_and_index_news(ParsedStock("삼전", "삼성전자", "005930"), "2020-01-01", "2020-01-03",
                               deadline=time.monotonic() - 1, **environment)
    environment["client"].embeddings.create.assert_not_called()


# 뉴스가 없는 질문부터 LLM 도구 선택·수집·인덱싱·재검색·최종 응답까지 전체 통합
def test_question_to_agent_collection_and_answer(environment, monkeypatch):
    from src.app import stock_advice_workflow as workflow
    from src.agents.stock_advisor_agent import decide_next_action
    from src.domain.advice_models import ParsedQuestion
    from src.domain.stock_models import DailyStock, StockTrend
    from src.search.hybrid_search import search_news_hybrid

    config = replace(environment["app_settings"], stock_api_provider="toss",
                     vector_db_path=str(environment["faiss_dir"]))
    environment["app_settings"] = config
    parsed = ParsedQuestion("SK하이닉스 전망", (ParsedStock("하이닉스", "SK하이닉스", "000660"),), "general_analysis")
    monkeypatch.setattr(workflow, "parse_stock_question", Mock(return_value=parsed))
    monkeypatch.setattr(workflow, "collect_stock_price", Mock(return_value=DailyStock(
        "SK하이닉스", "000660", "2020-01-03", 90, 110, 80, 100, 1000, 1, 1)))
    monkeypatch.setattr(workflow, "load_latest_stock_trend", Mock(return_value=StockTrend(
        "SK하이닉스", "000660", "2020-01-03", 5, 5, 3, 2, "최근 5거래일 상승")))
    monkeypatch.setattr(workflow, "_get_news_search_period", Mock(return_value=("2020-01-01", "2020-01-03")))
    monkeypatch.setattr(workflow, "BM25_TOKENS_PATH", environment["bm25_path"])
    client = environment["client"]

    # LLM 요청의 실제 상태에 따라 먼저 뉴스 수집, 다음에는 응답 결정
    def respond_agent(**options):
        data = json.loads(options["input"])
        missing = data["contexts"][0]["summary"]["news_count"] == 0
        decision = {"action": "collect_news" if missing else "respond", "reason": "뉴스 보충" if missing else "근거 확보",
                    "stock_code": "000660" if missing else None, "news_start": 1 if missing else None,
                    "search_query": None}
        return SimpleNamespace(status="completed", output=[SimpleNamespace(
            type="function_call", name="submit_agent_decision", arguments=json.dumps(decision))])

    client.responses.create.side_effect = respond_agent
    monkeypatch.setattr(workflow, "decide_next_action", lambda *args, **options:
                        decide_next_action(*args, client=client, **options))

    # 수집은 HTTP 대역으로, 인덱싱과 검색은 실제 구현으로 실행
    def collect(stock, start_date, end_date, **options):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_payload()))) as connection:
            merged = {**environment, **options, "http_client": connection}
            return collect_and_index_news(stock, start_date, end_date, **merged)

    monkeypatch.setattr(workflow, "collect_and_index_news", collect)
    monkeypatch.setattr(workflow, "search_news_hybrid", lambda question, **options: search_news_hybrid(
        question, client=client, kiwi=environment["kiwi"], bm25_path=environment["bm25_path"], **options))
    answer = workflow.run_stock_advice_workflow(parsed.original_question, config)
    assert "반도체 매출 증가" in answer
    assert "https://example.test/1" in answer
    assert "뉴스 근거가 부족" not in answer
    assert client.responses.create.call_count == 2
    assert client.embeddings.create.call_count == 2

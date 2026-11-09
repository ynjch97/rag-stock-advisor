from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config.settings import settings
from src.domain.news_models import NewsChunk
from src.indexing.faiss_index import build_faiss_index, save_faiss_index
from src.search.vector_search import search_news_vectors
from src.search import vector_search


# 실제 FAISS와 메타데이터를 만들고 OpenAI 응답은 대역으로 제공
@pytest.fixture
def search_environment(tmp_path):
    chunks = [NewsChunk(
        chunk_id=f"news-{i}_001", news_id=f"news-{i}", chunk_index=1, chunk_count=1,
        start_index=0, stock_name="삼성전자" if i != 1 else "SK하이닉스",
        stock_code="005930" if i != 1 else "000660", url=f"https://example.test/{i}",
        title=f"실적 기사 {i}", content=f"매출 증가 {i}", important="O", publisher="경제신문",
        published_date=f"2026-11-0{i + 1}",
    ) for i in range(3)]
    path = tmp_path / "stock_advisor.index"
    save_faiss_index(build_faiss_index([[3, 4], [4, 0], [0, 5]]), path)
    metadata = {
        "schema_version": 1, "configuration": {"metric": "cosine", "embedding_model": "test-model"},
        "embedding_model": "test-model", "dimension": 2, "chunk_count": 3,
        "chunks": [asdict(chunk) for chunk in chunks],
        "faiss_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta_path = tmp_path / "stock_advisor_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[10, 0])])
    config = replace(settings, openai_api_key=None, stock_api_key=None, stock_api_secret=None,
                     news_api_key=None, news_api_secret=None, vector_db_provider="faiss",
                     openai_embedding_model="test-model")
    return dict(faiss_dir=tmp_path, app_settings=config, client=client), meta_path


# BM25 없이 코사인 순위·정규화·메타데이터 매핑 및 저장 파일 보존 확인
def test_search_cosine_ranking_without_bm25(search_environment):
    kwargs, _ = search_environment
    before = {path.name: path.read_bytes() for path in kwargs["faiss_dir"].iterdir()}
    results = search_news_vectors(" 실적 증가 이유? ", top_k=10, **kwargs)
    assert [result["news_id"] for result in results] == ["news-1", "news-0", "news-2"]
    assert [result["score"] for result in results] == pytest.approx([1, 0.6, 0])
    assert results[0]["stock_code"] == "000660"
    assert results[0]["content"] == "매출 증가 1"
    assert kwargs["client"].embeddings.create.call_args.kwargs["input"] == ["실적 증가 이유?"]
    assert before == {path.name: path.read_bytes() for path in kwargs["faiss_dir"].iterdir()}


# 전체 순위에서 종목·기간 필터를 적용하여 상위 k개 누락 방지
def test_stock_and_inclusive_date_filters(search_environment):
    kwargs, _ = search_environment
    results = search_news_vectors("실적", top_k=1, stock_code="005930",
                                  start_date="2026-11-03", end_date="2026-11-03", **kwargs)
    assert [result["news_id"] for result in results] == ["news-2"]


# 조건에 맞는 기사가 없으면 유료 임베딩 요청 생략
def test_no_matching_chunks_skips_embedding(search_environment):
    kwargs, _ = search_environment
    assert search_news_vectors("실적", stock_code="999999", **kwargs) == []
    kwargs["client"].embeddings.create.assert_not_called()


# 임계 점수 포함 여부 검증
def test_minimum_score(search_environment):
    kwargs, _ = search_environment
    assert len(search_news_vectors("실적", min_score=1, **kwargs)) == 1
    assert len(search_news_vectors("실적", min_score=0.7, **kwargs)) == 1


# 잘못된 인자를 API 호출 전에 거부
@pytest.mark.parametrize("overrides", [
    {"question": " "}, {"question": None}, {"top_k": 0}, {"top_k": True}, {"top_k": 1.5},
    {"min_score": 1.1}, {"min_score": float("nan")}, {"min_score": True},
    {"stock_code": ""}, {"start_date": "invalid"},
    {"start_date": "2026-11-03", "end_date": "2026-11-01"},
])
def test_invalid_search_arguments(search_environment, overrides):
    kwargs, _ = search_environment
    arguments = dict(kwargs, question="실적")
    arguments.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        search_news_vectors(**arguments)
    kwargs["client"].embeddings.create.assert_not_called()


# 저장된 모델·개수·차원·매핑·체크섬 불일치는 API 요청 전에 거부
@pytest.mark.parametrize("failure", ["model", "count", "dimension", "duplicate", "fields", "checksum", "metric"])
def test_invalid_saved_index_stops_before_embedding(search_environment, failure):
    kwargs, path = search_environment
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if failure == "model":
        metadata["embedding_model"] = "other-model"
    elif failure == "count":
        metadata["chunk_count"] = 99
    elif failure == "dimension":
        metadata["dimension"] = 99
    elif failure == "duplicate":
        metadata["chunks"][1]["chunk_id"] = metadata["chunks"][0]["chunk_id"]
    elif failure == "fields":
        metadata["chunks"][0]["content"] = 123
    elif failure == "metric":
        metadata["configuration"]["metric"] = "l2"
    else:
        metadata["faiss_sha256"] = "incorrect"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        search_news_vectors("실적", **kwargs)
    kwargs["client"].embeddings.create.assert_not_called()


# 질문 벡터 차원 불일치·0벡터·NaN 거부
@pytest.mark.parametrize("vector", [[1, 2, 3], [0, 0], [float("nan"), 1]])
def test_invalid_question_embedding(search_environment, vector):
    kwargs, _ = search_environment
    kwargs["client"].embeddings.create.return_value.data[0].embedding = vector
    with pytest.raises(ValueError):
        search_news_vectors("실적", **kwargs)


# 인덱스 부재 시 수집·재구축이나 유료 임베딩을 실행하지 않음
def test_missing_index_does_not_call_embedding(search_environment, tmp_path):
    kwargs, _ = search_environment
    kwargs["faiss_dir"] = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        search_news_vectors("실적", **kwargs)
    kwargs["client"].embeddings.create.assert_not_called()


# 검색 전체를 지정 횟수만큼 호출하고 밀리초 기준 p50·p95를 출력
def test_benchmark_response_time(monkeypatch, capsys):
    search = Mock(return_value=[])
    monkeypatch.setattr(vector_search, "search_news_vectors", search)
    timestamps = iter([0, 0.1, 1, 1.2, 2, 2.3])
    monkeypatch.setattr(vector_search.time, "perf_counter", lambda: next(timestamps))
    result = vector_search.benchmark_news_vector_search("실적", repeats=3, top_k=2)
    assert result["p50_ms"] == pytest.approx(200)
    assert result["p95_ms"] == pytest.approx(290)
    assert result["repeats"] == 3
    assert search.call_count == 3
    search.assert_called_with("실적", top_k=2)
    assert "p50 : 200.000 ms, p95 : 290.000 ms (n=3)" in capsys.readouterr().out


# 반복 횟수가 잘못되면 검색·API 호출 전에 거부
@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_benchmark_invalid_repeat_count(monkeypatch, repeats):
    search = Mock()
    monkeypatch.setattr(vector_search, "search_news_vectors", search)
    with pytest.raises(ValueError):
        vector_search.benchmark_news_vector_search("실적", repeats=repeats)
    search.assert_not_called()

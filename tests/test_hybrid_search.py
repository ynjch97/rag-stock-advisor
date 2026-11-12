from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config.settings import settings
from src.domain.news_models import NewsChunk
from src.indexing.news_indexer import index_news
from src.search import hybrid_search
from src.search.hybrid_search import combine_news_search_results, search_news_hybrid


# 두 검색이 공통으로 반환하는 청크 정보에 검색별 원점수만 추가
def _result(chunk_id, score, important="O"):
    return {"chunk_id": chunk_id, "important": important, "content": f"본문 {chunk_id}", "score": score}


# 독립 Min-Max 정규화·가중 합산·ID 합집합·미등장 점수와 원본 보존
def test_normalization_weighted_sum_and_union():
    vectors = [_result("a", 0.2), _result("b", 0.6), _result("c", 1.0)]
    keywords = [_result("b", 10), _result("c", 20), _result("d", 30)]
    original = deepcopy((vectors, keywords))
    results = combine_news_search_results(vectors, keywords)
    assert [item["chunk_id"] for item in results] == ["c", "b", "d", "a"]
    by_id = {item["chunk_id"]: item for item in results}
    assert by_id["c"]["hybrid_score"] == pytest.approx(0.85)
    assert by_id["b"]["hybrid_score"] == pytest.approx(0.35)
    assert by_id["d"]["hybrid_score"] == pytest.approx(0.3)
    assert by_id["a"]["vector_score"] == 0.2
    assert by_id["a"]["keyword_score"] is None
    assert by_id["d"]["vector_score"] is None
    assert by_id["c"]["vector_normalized_score"] == 1
    assert by_id["c"]["keyword_normalized_score"] == 0.5
    assert "score" not in by_id["c"]
    assert (vectors, keywords) == original


# 단일·전부 동점 후보는 1, 빈 검색 쪽은 0이며 가중치를 재분배하지 않음
@pytest.mark.parametrize("vectors, keywords, expected", [
    ([], [], []),
    ([_result("a", 0)], [], [0.7]),
    ([], [_result("a", -2)], [0.3]),
    ([_result("a", 0), _result("b", 0)], [], [0.7, 0.7]),
    ([_result("a", 0)], [_result("a", -2)], [1.0]),
])
def test_empty_single_and_equal_scores(vectors, keywords, expected):
    results = combine_news_search_results(vectors, keywords)
    assert [item["hybrid_score"] for item in results] == pytest.approx(expected)


# 음수 BM25 점수도 크기순으로 0~1 정규화
def test_negative_bm25_scores():
    results = combine_news_search_results([], [_result("a", -4), _result("b", -2)])
    assert [item["chunk_id"] for item in results] == ["b", "a"]
    assert results[0]["keyword_normalized_score"] == 1
    assert results[0]["keyword_score"] == -2


# 유효하지 않은 원점수·중복 ID는 결합 전에 거부
@pytest.mark.parametrize("results", [
    [_result("a", float("nan"))], [_result("a", float("inf"))], [_result("a", True)],
    [_result("a", None)], [_result("", 1)], [_result("a", 1), _result("a", 2)],
])
def test_invalid_search_results(results):
    with pytest.raises(ValueError):
        combine_news_search_results(results, [])


# 같은 ID인데 본문·메타데이터가 다르면 서로 다른 인덱스 결과 혼합 방지
def test_conflicting_chunk_metadata():
    with pytest.raises(ValueError, match="different metadata"):
        combine_news_search_results([_result("a", 0.5)], [_result("a", 2, important="X")])


# 양쪽 후보 개수·필터·저장 경로 전달 및 합산 후 중요도 재정렬 확인
def test_search_orchestration(monkeypatch, tmp_path):
    vector = Mock(return_value=[_result("a", 0.9, "X"), _result("b", 0.1)])
    keyword = Mock(return_value=[_result("a", 2, "X"), _result("b", 1)])
    monkeypatch.setattr(hybrid_search, "search_news_vectors", vector)
    monkeypatch.setattr(hybrid_search, "search_news_keywords", keyword)
    client, kiwi = Mock(), Mock()
    results = search_news_hybrid("실적", top_k=1, stock_code="005930", start_date="2026-11-01",
                                 end_date="2026-11-05", faiss_dir=tmp_path, client=client, kiwi=kiwi)
    assert results[0]["chunk_id"] == "b"
    assert results[0]["hybrid_score"] == 0
    for search in (vector, keyword):
        assert search.call_args.kwargs["top_k"] == 20
        assert search.call_args.kwargs["stock_code"] == "005930"
        assert search.call_args.kwargs["start_date"] == "2026-11-01"
        assert search.call_args.kwargs["end_date"] == "2026-11-05"
    assert keyword.call_args.kwargs["metadata_path"] == tmp_path / "stock_advisor_metadata.json"
    assert vector.call_args.kwargs["client"] is client
    assert keyword.call_args.kwargs["kiwi"] is kiwi


# 설정 경로와 후보 개수를 기존 검색 함수에 그대로 전달
def test_custom_candidate_count_and_settings_path(monkeypatch, tmp_path):
    vector, keyword = Mock(return_value=[]), Mock(return_value=[])
    monkeypatch.setattr(hybrid_search, "search_news_vectors", vector)
    monkeypatch.setattr(hybrid_search, "search_news_keywords", keyword)
    config = replace(settings, vector_db_path=str(tmp_path))
    assert search_news_hybrid("실적", top_k=2, candidate_k=10, app_settings=config) == []
    assert vector.call_args.kwargs["top_k"] == 10
    assert vector.call_args.kwargs["faiss_dir"] == tmp_path


# 잘못된 후보·최종 개수는 API 요청 전에 거부
@pytest.mark.parametrize("options", [
    {"top_k": 0}, {"top_k": True}, {"candidate_k": 0}, {"candidate_k": True},
    {"candidate_k": 1.5}, {"top_k": 6, "candidate_k": 5},
])
def test_invalid_counts(monkeypatch, options):
    vector = Mock()
    monkeypatch.setattr(hybrid_search, "search_news_vectors", vector)
    with pytest.raises(ValueError):
        search_news_hybrid("실적", **options)
    vector.assert_not_called()


# 로컬 인덱스 오류는 빈 결과로 숨기지 않고 유료 임베딩 전에 전파
def test_missing_keyword_index_stops_before_embedding(monkeypatch):
    vector = Mock()
    monkeypatch.setattr(hybrid_search, "search_news_vectors", vector)
    monkeypatch.setattr(hybrid_search, "search_news_keywords", Mock(side_effect=FileNotFoundError))
    with pytest.raises(FileNotFoundError):
        search_news_hybrid("실적")
    vector.assert_not_called()


# 실제 인덱서 산출물로 FAISS·BM25·정규화·재정렬 통합 검증; API만 대역 사용
def test_hybrid_with_saved_indexes(tmp_path):
    chunks = [NewsChunk(
        chunk_id=f"news-{i}_001", news_id=f"news-{i}", chunk_index=1, chunk_count=1,
        start_index=0, stock_name="삼성전자", stock_code="005930", url=f"https://example.test/{i}",
        title=f"기사{i}", content=content, important="O" if i == 1 else "X",
        publisher="경제신문", published_date="2026-11-01",
    ) for i, content in enumerate(["반도체 매출", "반도체 실적", "자동차 공장", "로봇 투자", "바이오 임상"])]
    directory = tmp_path / "chunks"
    directory.mkdir()
    (directory / "005930_news_20261101_100000_000001_chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks]), encoding="utf-8",
    )
    kiwi = Mock()
    kiwi.tokenize.side_effect = lambda text: [SimpleNamespace(form=word, tag="NNG") for word in text.split()]
    client = Mock()
    client.embeddings.create.side_effect = lambda **options: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[1, float(i + 1)]) for i in range(len(options["input"]))
    ])
    config = replace(settings, openai_api_key=None, stock_api_key=None, stock_api_secret=None,
                     news_api_key=None, news_api_secret=None, vector_db_provider="faiss",
                     openai_embedding_model="test-model")
    paths = dict(faiss_dir=tmp_path / "faiss", bm25_path=tmp_path / "tokens.json")
    index_news(directory, app_settings=config, client=client, kiwi=kiwi, **paths)
    results = search_news_hybrid("반도체", app_settings=config, client=client, kiwi=kiwi, **paths)
    assert len(results) == 5
    assert results[0]["chunk_id"] == "news-1_001"
    assert all(0 <= result["hybrid_score"] <= 1 for result in results)
    assert len({result["chunk_id"] for result in results}) == 5
    assert any(result["keyword_score"] is None for result in results)
    assert client.embeddings.create.call_count == 2

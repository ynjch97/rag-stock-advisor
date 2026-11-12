from copy import deepcopy

import pytest

from src.search.search_result_ranker import rerank_news_results


# 중요도 O를 우선하고 같은 중요도는 합산 점수·청크 ID순으로 정렬
def test_importance_then_score_and_tie_order():
    results = [
        {"chunk_id": "x", "important": "X", "hybrid_score": 1.0, "vector_score": 0.9},
        {"chunk_id": "b", "important": "O", "hybrid_score": 0.2, "keyword_score": -0.5},
        {"chunk_id": "a", "important": "O", "hybrid_score": 0.2},
        {"chunk_id": "c", "important": "O", "hybrid_score": 0.8},
    ]
    original = deepcopy(results)
    ranked = rerank_news_results(results, top_k=3)
    assert [result["chunk_id"] for result in ranked] == ["c", "a", "b"]
    assert ranked[-1]["keyword_score"] == -0.5
    assert results == original


# 빈 검색 결과는 빈 목록으로 유지
def test_empty_results():
    assert rerank_news_results([]) == []


# 잘못된 반환 개수와 점수·메타데이터 거부
@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        rerank_news_results([], top_k=top_k)


# 정규화 범위를 벗어난 점수와 불완전한 결과 거부
@pytest.mark.parametrize("update", [
    {"chunk_id": ""}, {"important": "Y"}, {"hybrid_score": None},
    {"hybrid_score": True}, {"hybrid_score": float("nan")},
    {"hybrid_score": -0.1}, {"hybrid_score": 1.1},
])
def test_invalid_result(update):
    result = dict(chunk_id="a", important="O", hybrid_score=0.5)
    result.update(update)
    with pytest.raises(ValueError):
        rerank_news_results([result])

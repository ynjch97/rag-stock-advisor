from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config.settings import settings
from src.domain.news_models import NewsChunk
from src.indexing.bm25_index import load_bm25_index, save_bm25_index, tokenize_text
from src.search.keyword_search import search_news_keywords


# 토큰 변경 테스트에서도 체크섬은 실제 파일 내용에 맞게 갱신
def _write_tokens(kwargs, tokens):
    save_bm25_index(tokens, [f"news-{i}_001" for i in range(5)], kwargs["bm25_path"], fingerprint="snapshot")
    path = kwargs["metadata_path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["bm25_sha256"] = hashlib.sha256(kwargs["bm25_path"].read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata), encoding="utf-8")


# FAISS 바이너리와 API 키 없이 BM25 JSON·청크 매핑만 준비
@pytest.fixture
def keyword_environment(tmp_path):
    chunks = [NewsChunk(
        chunk_id=f"news-{i}_001", news_id=f"news-{i}", chunk_index=1, chunk_count=1,
        start_index=0, stock_name="삼성전자" if i % 2 == 0 else "SK하이닉스",
        stock_code="005930" if i % 2 == 0 else "000660", url=f"https://example.test/{i}",
        title=f"기사 {i}", content=f"본문 {i}", important="X" if i == 0 else "O",
        publisher="경제신문", published_date=f"2026-11-0{i + 1}",
    ) for i in range(5)]
    path = tmp_path / "stock_advisor_metadata.json"
    path.write_text(json.dumps({"schema_version": 1, "fingerprint": "snapshot", "chunk_count": 5,
                                "chunks": [asdict(chunk) for chunk in chunks]}), encoding="utf-8")
    kiwi = Mock()
    kiwi.tokenize.side_effect = lambda text: [SimpleNamespace(form=word, tag="NNG") for word in text.split()]
    kwargs = dict(bm25_path=tmp_path / "tokens.json", metadata_path=path, kiwi=kiwi)
    _write_tokens(kwargs, [["반도체", "반도체", "매출"], ["반도체", "매출"],
                           ["자동차", "공장"], ["로봇", "투자"], ["바이오", "임상"]])
    return kwargs


# 점수 내림차순과 청크 매핑·기본 top_k 확인; 중요도는 순위에 반영하지 않음
def test_keyword_ranking_and_metadata(keyword_environment):
    kwargs = keyword_environment
    before = {path.name: path.read_bytes() for path in kwargs["bm25_path"].parent.iterdir()}
    results = search_news_keywords("반도체", top_k=10, **kwargs)
    assert [result["news_id"] for result in results] == ["news-0", "news-1"]
    bm25, _ = load_bm25_index(kwargs["bm25_path"])
    assert [result["score"] for result in results] == pytest.approx(bm25.get_scores(["반도체"])[:2])
    assert results[0]["important"] == "X"
    assert results[0]["content"] == "본문 0"
    assert len(search_news_keywords("반도체", top_k=1, **kwargs)) == 1
    assert before == {path.name: path.read_bytes() for path in kwargs["bm25_path"].parent.iterdir()}


# 종목 및 양 끝 날짜를 포함하는 기간 필터 적용 후 상위 결과 선택
def test_keyword_stock_date_filters(keyword_environment):
    results = search_news_keywords("반도체 자동차", top_k=1, stock_code="005930",
                                  start_date="2026-11-03", end_date="2026-11-03", **keyword_environment)
    assert [result["news_id"] for result in results] == ["news-2"]


# 일치 단어 또는 해당 종목·기간 데이터가 없으면 빈 목록
@pytest.mark.parametrize("options", [
    {"question": "없는단어"}, {"stock_code": "999999"}, {"start_date": "2027-01-01"},
])
def test_keyword_no_results(keyword_environment, options):
    arguments = dict(keyword_environment, question="반도체")
    arguments.update(options)
    assert search_news_keywords(**arguments) == []


# 임계값은 포함 비교이며 코사인처럼 -1~1로 제한하지 않음
def test_keyword_minimum_score(keyword_environment):
    results = search_news_keywords("반도체", **keyword_environment)
    assert len(search_news_keywords("반도체", min_score=results[0]["score"], **keyword_environment)) == 1
    assert search_news_keywords("반도체", min_score=100, **keyword_environment) == []


# IDF가 0·음수인 실제 일치 기사도 기본 검색에서는 반환
@pytest.mark.parametrize("tokens, expected_count", [
    ([["공통"]] * 5, 5), ([["공통"]] * 4 + [["기타"]], 4),
])
def test_nonpositive_matching_scores_are_preserved(keyword_environment, tokens, expected_count):
    _write_tokens(keyword_environment, tokens)
    results = search_news_keywords("공통", **keyword_environment)
    assert len(results) == expected_count
    assert all(result["score"] <= 0 for result in results)


# 저장·질문에 동일한 영문 대소문자 통일 정책 적용
def test_keyword_casefold(keyword_environment):
    _write_tokens(keyword_environment, [["sk"]] + [["다른단어"]] * 4)
    assert search_news_keywords("SK", **keyword_environment)[0]["news_id"] == "news-0"


# 실제 Kiwi로 구축한 토큰도 동일한 기본 토큰화로 검색 가능
def test_keyword_real_kiwi(keyword_environment):
    kwargs = dict(keyword_environment)
    kwargs.pop("kiwi")
    _write_tokens(kwargs, [tokenize_text(text) for text in [
        "삼성전자 반도체 매출", "하이닉스 반도체 실적", "자동차 공장", "로봇 투자", "바이오 임상",
    ]])
    results = search_news_keywords("반도체", **kwargs)
    assert {result["news_id"] for result in results} == {"news-0", "news-1"}


# 문장부호만 남아 토큰이 없으면 인덱스 로딩 없이 빈 결과
def test_punctuation_only_query(keyword_environment):
    keyword_environment["kiwi"].tokenize.return_value = []
    keyword_environment["kiwi"].tokenize.side_effect = None
    assert search_news_keywords("...", **keyword_environment) == []


# 잘못된 인자 검증
@pytest.mark.parametrize("options", [
    {"question": " "}, {"question": None}, {"top_k": 0}, {"top_k": True}, {"top_k": 1.5},
    {"min_score": float("nan")}, {"min_score": float("inf")}, {"min_score": True},
    {"stock_code": ""}, {"start_date": "invalid"},
    {"start_date": "2026-11-05", "end_date": "2026-11-01"},
])
def test_keyword_invalid_arguments(keyword_environment, options):
    arguments = dict(keyword_environment, question="반도체")
    arguments.update(options)
    with pytest.raises((ValueError, TypeError)):
        search_news_keywords(**arguments)


# 메타데이터 오류·다른 인덱스 세대·순서 불일치를 거부
@pytest.mark.parametrize("failure", ["checksum", "fingerprint", "count", "order", "duplicate", "field", "schema"])
def test_keyword_invalid_metadata(keyword_environment, failure):
    path = keyword_environment["metadata_path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if failure == "checksum":
        metadata["bm25_sha256"] = "incorrect"
    elif failure == "fingerprint":
        metadata["fingerprint"] = "other-snapshot"
    elif failure == "count":
        metadata["chunk_count"] = 100
    elif failure == "order":
        metadata["chunks"].reverse()
    elif failure == "duplicate":
        metadata["chunks"][1]["chunk_id"] = metadata["chunks"][0]["chunk_id"]
    elif failure == "schema":
        metadata["schema_version"] = 99
    else:
        metadata["chunks"][0]["content"] = None
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        search_news_keywords("반도체", **keyword_environment)


# 인덱스가 없으면 파일 생성·수집 없이 오류 반환
def test_keyword_missing_file(keyword_environment):
    keyword_environment["bm25_path"].unlink()
    with pytest.raises(FileNotFoundError):
        search_news_keywords("반도체", **keyword_environment)
    assert not keyword_environment["bm25_path"].exists()


# 기본 메타데이터 경로는 기존 앱 설정의 인덱스 디렉터리 사용
def test_keyword_default_metadata_path(keyword_environment):
    kwargs = dict(keyword_environment)
    directory = kwargs.pop("metadata_path").parent
    kwargs["app_settings"] = replace(settings, vector_db_path=str(directory))
    assert search_news_keywords("반도체", **kwargs)

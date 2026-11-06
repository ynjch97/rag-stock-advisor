from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
from openai import APIConnectionError
import pytest

from src.config.settings import settings
from src.domain.news_models import NewsChunk
from src.indexing import news_indexer as indexer


# 같은 기사에 속한 연속된 청크 생성
def _article(news_id, url, count=1, stock_code="005930"):
    return [NewsChunk(
        chunk_id=f"{news_id}_{position:03d}", news_id=news_id, chunk_index=position,
        chunk_count=count, start_index=(position - 1) * 10, stock_name="삼성전자",
        stock_code=stock_code, url=url, title=f"실적 {news_id}", content=f"매출 {position}조원 {news_id}",
        important="O", publisher="테스트언론사", published_date="2026-11-05",
    ) for position in range(1, count + 1)]


# 실제 청킹 스냅샷의 파일명 형식과 JSON 구조로 저장
def _snapshot(directory, timestamp, chunks):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"005930_news_{timestamp}_chunks.json"
    path.write_text(json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False), encoding="utf-8")
    return path


# 모델 API를 호출하지 않는 SDK·Kiwi 대역과 임시 저장 경로 제공
@pytest.fixture
def environment(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _snapshot(chunks_dir, "20261105_100000_000001", _article("a-old", "https://example.test/a", 2)
              + _article("d", "https://example.test/d") + _article("e", "https://example.test/e"))
    latest = _snapshot(chunks_dir, "20261106_100000_000001", _article("a-new", "https://example.test/a")
                       + _article("b", "https://example.test/b") + _article("c", "https://example.test/c"))
    client = Mock()

    # API 반환 벡터를 요청 위치와 문서 내용에 따라 결정적으로 생성
    def embeddings(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(
            index=index, embedding=[1.0, float(sum(map(ord, text)) % 17 + 1), float(len(text))],
        ) for index, text in enumerate(kwargs["input"])])

    client.embeddings.create.side_effect = embeddings
    kiwi = Mock()
    kiwi.tokenize.side_effect = lambda text: [SimpleNamespace(form=word, tag="NNG") for word in text.split()]
    config = replace(settings, openai_api_key=None, stock_api_key=None, stock_api_secret=None,
                     news_api_key=None, news_api_secret=None, vector_db_provider="faiss",
                     openai_embedding_model="test-embedding")
    kwargs = dict(app_settings=config, faiss_dir=tmp_path / "faiss", bm25_path=tmp_path / "bm25" / "tokens.json",
                  client=client, kiwi=kiwi)
    return chunks_dir, latest, kwargs


# 이전 파일 속 A 기사도 중복으로 인식하고 최신 A의 청크 전체로 교체
def test_latest_article_chunks_are_selected_as_a_whole(environment):
    directory, _, _ = environment
    chunks = indexer.load_latest_news_chunks(directory)
    assert [chunk.news_id for chunk in chunks] == ["a-new", "b", "c", "d", "e"]
    assert len(chunks) == 5
    assert all(chunk.news_id != "a-old" for chunk in chunks)


# 파일 복사로 수정 시각이 바뀌어도 파일명에 기록된 실제 수집 시각 우선
def test_snapshot_timestamp_takes_priority_over_mtime(environment):
    directory, latest, _ = environment
    old = next(path for path in directory.iterdir() if path != latest)
    os.utime(old, (2_000_000_000, 2_000_000_000))
    os.utime(latest, (1, 1))
    assert indexer.load_latest_news_chunks(directory)[0].news_id == "a-new"


# 모든 단계가 같은 청크 순서를 사용하고 실제 FAISS·BM25를 저장·재로딩
def test_indexing_end_to_end_and_unchanged_reuse(environment):
    directory, _, kwargs = environment
    result = indexer.index_news(directory, batch_size=2, **kwargs)
    assert result["rebuilt"] is True and result["chunk_count"] == 5 and result["dimension"] == 3
    client, kiwi = kwargs["client"], kwargs["kiwi"]
    assert client.embeddings.create.call_count == 3
    requested_texts = [text for call in client.embeddings.create.call_args_list for text in call.kwargs["input"]]
    assert requested_texts == [call.args[0] for call in kiwi.tokenize.call_args_list]
    faiss, bm25, metadata = indexer.load_news_indexes(kwargs["faiss_dir"], kwargs["bm25_path"])
    assert faiss.ntotal == bm25.corpus_size == len(metadata["chunks"]) == 5
    assert metadata["embedding_model"] == "test-embedding"
    assert [record["news_id"] for record in metadata["chunks"]] == ["a-new", "b", "c", "d", "e"]
    query = faiss.reconstruct(0).reshape(1, -1)
    scores, rows = faiss.search(query, 1)
    assert 0 <= int(rows[0][0]) < 5
    assert np.isclose(scores[0][0], 1.0)
    before = _artifact_bytes(kwargs)
    again = indexer.index_news(directory, **kwargs)
    assert again["rebuilt"] is False
    assert client.embeddings.create.call_count == 3
    assert _artifact_bytes(kwargs) == before


# 임베딩 모델·본문 변경은 전체 재구축을 유도
@pytest.mark.parametrize("change", ["content", "model"])
def test_changed_input_rebuilds_indexes(environment, change):
    directory, latest, kwargs = environment
    indexer.index_news(directory, **kwargs)
    if change == "content":
        records = json.loads(latest.read_text(encoding="utf-8"))
        records[0]["content"] = "정정된 매출 100조원"
        latest.write_text(json.dumps(records), encoding="utf-8")
    else:
        kwargs["app_settings"] = replace(kwargs["app_settings"], openai_embedding_model="other-model")
    assert indexer.index_news(directory, **kwargs)["rebuilt"] is True
    assert kwargs["client"].embeddings.create.call_count == 2


# 토큰화 정책 변경 시 이전 설정의 로딩 검증 대신 새 인덱스 구축
def test_tokenizer_configuration_change_rebuilds_indexes(environment, monkeypatch):
    directory, _, kwargs = environment
    indexer.index_news(directory, **kwargs)
    monkeypatch.setitem(indexer.TOKENIZER_SPEC, "policy", "surface-v2")
    assert indexer.index_news(directory, **kwargs)["rebuilt"] is True
    assert kwargs["client"].embeddings.create.call_count == 2
    indexer.load_news_indexes(kwargs["faiss_dir"], kwargs["bm25_path"])


# 유료 API 요청 실패 시 기존 정상 인덱스 3개는 변경하지 않음
def test_embedding_failure_keeps_existing_indexes(environment):
    directory, _, kwargs = environment
    indexer.index_news(directory, **kwargs)
    before = _artifact_bytes(kwargs)
    kwargs["client"].embeddings.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    with pytest.raises(ValueError, match="OpenAI embedding"):
        indexer.index_news(directory, force=True, **kwargs)
    assert _artifact_bytes(kwargs) == before
    indexer.load_news_indexes(kwargs["faiss_dir"], kwargs["bm25_path"])


# 파일 교체 중 두 번째 교체가 실패해도 먼저 교체된 FAISS까지 원래 상태로 복구
def test_publish_failure_rolls_back_all_artifacts(environment, monkeypatch):
    directory, _, kwargs = environment
    indexer.index_news(directory, **kwargs)
    before = _artifact_bytes(kwargs)
    original_replace = Path.replace

    # BM25 교체만 실패시키고 복구 작업은 실제 파일 시스템에서 수행
    def fail_bm25_replace(path, target):
        if path.name == "tokens.json" and target == kwargs["bm25_path"]:
            raise OSError("simulated publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_bm25_replace)
    with pytest.raises(OSError, match="publication"):
        indexer.index_news(directory, force=True, **kwargs)
    assert _artifact_bytes(kwargs) == before
    indexer.load_news_indexes(kwargs["faiss_dir"], kwargs["bm25_path"])


# 토큰 파일이 다른 세대의 데이터와 섞이면 조용히 사용하지 않음
def test_corrupt_artifact_is_not_reused(environment):
    directory, _, kwargs = environment
    indexer.index_news(directory, **kwargs)
    kwargs["bm25_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        indexer.index_news(directory, **kwargs)
    assert kwargs["client"].embeddings.create.call_count == 1
    assert indexer.index_news(directory, force=True, **kwargs)["rebuilt"]


# 새 기사 스냅샷이 불완전·충돌 ID이면 임베딩 호출 전에 거부
@pytest.mark.parametrize("failure", ["incomplete", "collision", "mixed", "field"])
def test_invalid_news_chunks_stop_before_embedding(environment, failure):
    directory, latest, kwargs = environment
    records = json.loads(latest.read_text(encoding="utf-8"))
    if failure == "incomplete":
        records[0]["chunk_count"] = 2
    elif failure == "collision":
        records[1]["chunk_id"] = records[0]["chunk_id"]
    elif failure == "mixed":
        duplicate = dict(records[0], news_id="other-a", chunk_id="other-a-1")
        records.append(duplicate)
    else:
        records[0]["content"] = None
    latest.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        indexer.index_news(directory, **kwargs)
    kwargs["client"].embeddings.create.assert_not_called()


# 청킹 파일이 없으면 정상 인덱스를 빈 결과로 덮어쓰지 않음
def test_empty_corpus_does_not_overwrite_existing(environment, tmp_path):
    directory, _, kwargs = environment
    indexer.index_news(directory, **kwargs)
    before = _artifact_bytes(kwargs)
    with pytest.raises(ValueError, match="No news chunks"):
        indexer.index_news(tmp_path / "empty", **kwargs)
    assert _artifact_bytes(kwargs) == before


# 실제 산출물만 비교하여 임시 파일 정리 여부와 무관하게 기존 파일 보존 검증
def _artifact_bytes(kwargs):
    return tuple(path.read_bytes() for path in (
        kwargs["faiss_dir"] / "stock_advisor.index",
        kwargs["faiss_dir"] / "stock_advisor_metadata.json", kwargs["bm25_path"],
    ))

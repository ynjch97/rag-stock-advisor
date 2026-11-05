from dataclasses import asdict, replace
import json

import pytest

from src.chunking import document_chunker as chunker
from src.config import paths
from src.domain.news_models import NewsChunk, ProcessedNews


# API·실제 저장 데이터에 의존하지 않는 전처리 기사 제공
@pytest.fixture
def news():
    return ProcessedNews(
        news_id="261106123045_01", stock_name="삼성전자", stock_code="005930",
        url="https://example.test/news/1", title="삼성전자 실적 전망",
        content="삼성전자 영업이익은 10조원이다. 매출은 감소하지 않았다.",
        important="O", publisher="테스트언론사", published_date="2026-11-05",
    )


# 짧은 기사와 요약문은 분할·요약·공백 변경 없이 하나로 유지
@pytest.mark.parametrize("size", [1, 500, 799, 800])
def test_short_article_keeps_entire_content_and_metadata(news, size):
    news = replace(news, content="가" * size)
    chunks = chunker.chunk_news(news)
    assert len(chunks) == 1
    result = chunks[0]
    assert isinstance(result, NewsChunk)
    assert result.content == news.content
    assert result.chunk_id == "261106123045_01_001"
    assert (result.chunk_index, result.chunk_count, result.start_index) == (1, 1, 0)
    for key in ("news_id", "stock_name", "stock_code", "title", "important", "publisher", "published_date", "url"):
        assert getattr(result, key) == getattr(news, key)


# 문자 분할이 필요한 본문에서도 800자·100자 overlap과 전체 범위 보존
def test_long_text_uses_target_size_and_overlap(news):
    content = "".join(chr(0x4E00 + index) for index in range(1800))
    chunks = chunker.chunk_news(replace(news, content=content))
    assert len(chunks) == 3
    assert all(0 < len(chunk.content) <= 800 for chunk in chunks)
    assert chunks[0].content[-100:] == chunks[1].content[:100]
    assert chunks[1].content[-100:] == chunks[2].content[:100]
    assert [chunk.start_index for chunk in chunks] == [0, 700, 1400]
    assert chunks[0].content + chunks[1].content[100:] + chunks[2].content[100:] == content
    assert [chunk.chunk_index for chunk in chunks] == [1, 2, 3]
    assert all(chunk.chunk_count == 3 for chunk in chunks)


# 문단·문장·금융 수치·부정 표현이 있는 실제 형태의 본문을 수정 없이 분할
def test_paragraphs_and_financial_expressions_are_preserved(news):
    content = "\n\n".join(
        f"{index}번째 문단. 삼성전자 영업이익은 10,000억원, 등락률은 -3.5%이다. 감소하지 않았다. " * 4
        for index in range(15)
    )
    chunks = chunker.chunk_news(replace(news, content=content))
    assert len(chunks) > 1
    rebuilt = ""
    cursor = 0
    for chunk in chunks:
        start, end = chunk.start_index, chunk.start_index + len(chunk.content)
        assert chunk.content == content[start:end]
        assert start <= cursor
        rebuilt += content[max(cursor, start):end]
        cursor = max(cursor, end)
    assert rebuilt == content
    assert any("10,000억원" in chunk.content and "-3.5%" in chunk.content for chunk in chunks)
    assert any("감소하지 않았다" in chunk.content for chunk in chunks)


# 강제 문자 경계에 걸린 금융 수치·단위는 잘라 두 조각으로 저장하지 않음
@pytest.mark.parametrize("amount", ["-3.5%", "10,000억원", "₩1,350", "+12.75달러", "10 만원"])
def test_chunk_boundary_does_not_split_financial_amount(news, amount):
    content = "가" * 798 + amount + "나" * 900
    chunks = chunker.chunk_news(replace(news, content=content))
    assert any(amount in chunk.content for chunk in chunks)
    spans = [match.span() for match in chunker.FINANCIAL_NUMBER.finditer(content)]
    for chunk in chunks:
        start, end = chunk.start_index, chunk.start_index + len(chunk.content)
        assert not any(left < start < right or left < end < right for left, right in spans)
        assert chunk.content == content[start:end]


# 같은 입력으로 청킹하면 같은 순서·ID·본문 생성
def test_chunk_ids_are_reproducible(news):
    news = replace(news, content=(news.content + "\n") * 50)
    first, second = chunker.chunk_news(news), chunker.chunk_news(news)
    assert first == second
    assert len({chunk.chunk_id for chunk in first}) == len(first)


# 정수가 아닌 설정·잘못된 범위는 분할 전에 거부
@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (800, -1), (800, 800), (True, 0), (800, 1.5)])
def test_invalid_chunk_settings(news, size, overlap):
    with pytest.raises(ValueError):
        chunker.chunk_news(news, chunk_size=size, chunk_overlap=overlap)


# 본문·필수 메타데이터 누락 및 중요도 오류는 정상 기사로 취급하지 않음
@pytest.mark.parametrize("field,value", [("content", " "), ("news_id", ""), ("title", None),
                                        ("published_date", "bad"), ("important", "Y")])
def test_invalid_news_is_rejected(news, field, value):
    with pytest.raises(ValueError):
        chunker.chunk_news(replace(news, **{field: value}))


# 여러 기사라도 내용은 섞지 않고 각각의 ID·메타데이터를 유지하며 스냅샷 저장
def test_chunk_and_save_preserves_processed_snapshot(tmp_path, news, monkeypatch):
    source = tmp_path / "processed" / "005930_news_20261106_123045.json"
    source.parent.mkdir()
    second = replace(news, news_id="261106123045_02", stock_name="SK하이닉스", stock_code="000660",
                     content="SK하이닉스 관련 기사이다.", important="X")
    source.write_text(json.dumps([asdict(news), asdict(second)], ensure_ascii=False), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setattr(paths, "NEWS_CHUNKS_DIR", tmp_path / "chunks")
    chunks = chunker.chunk_and_save_news(source)
    destination = tmp_path / "chunks" / "005930_news_20261106_123045_chunks.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == [asdict(chunk) for chunk in chunks]
    assert source.read_bytes() == before
    assert [chunk.news_id for chunk in chunks] == [news.news_id, second.news_id]
    assert [chunk.content for chunk in chunks] == [news.content, second.content]
    assert [chunk.important for chunk in chunks] == ["O", "X"]
    with pytest.raises(ValueError, match="must not overwrite processed"):
        chunker.chunk_and_save_news(source, source)


# 서로 다른 수집 스냅샷의 청킹 결과를 덮어쓰지 않음
def test_different_snapshots_use_different_output_paths(tmp_path, news, monkeypatch):
    monkeypatch.setattr(paths, "NEWS_CHUNKS_DIR", tmp_path / "chunks")
    for name in ("first.json", "second.json"):
        source = tmp_path / name
        source.write_text(json.dumps([asdict(news)]), encoding="utf-8")
        chunker.chunk_and_save_news(source)
    assert {path.name for path in (tmp_path / "chunks").iterdir()} == {"first_chunks.json", "second_chunks.json"}


# 손상된 JSON 구조·구형 모델·중복 ID는 기존 결과를 부분 덮어쓰지 않음
@pytest.mark.parametrize("failure", ["object", "old_model", "duplicate", "missing_content"])
def test_bad_processed_data_preserves_existing_output(tmp_path, news, failure):
    record = asdict(news)
    if failure == "old_model":
        record["summary"] = record.pop("content")
    elif failure == "missing_content":
        record["content"] = None
    records = record if failure == "object" else [record, record] if failure == "duplicate" else [record]
    source, destination = tmp_path / "processed.json", tmp_path / "chunks.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        chunker.chunk_and_save_news(source, destination)
    assert destination.read_text(encoding="utf-8") == "existing"


# 조회 결과가 비어 있으면 청킹 결과도 빈 배열로 저장
def test_empty_processed_snapshot(tmp_path):
    source, destination = tmp_path / "processed.json", tmp_path / "chunks.json"
    source.write_text("[]", encoding="utf-8")
    assert chunker.chunk_and_save_news(source, destination) == []
    assert json.loads(destination.read_text(encoding="utf-8")) == []

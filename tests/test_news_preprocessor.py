from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json

import pytest

from src.domain.news_models import ProcessedNews
from src.preprocessing import news_preprocessor as preprocessor
from src.preprocessing.text_cleaner import clean_news_text


# 실제 현재 시각 대신 고정된 한국 시각으로 초 단위 ID 생성 검증
@pytest.fixture(autouse=True)
def preprocessing_clock(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat("2026-11-06T12:30:45+09:00").astimezone(tz)

    monkeypatch.setattr(preprocessor, "datetime", FixedDatetime)


# 실제 수집 구조와 같은 모의 raw 스냅샷 제공
@pytest.fixture
def raw_news():
    return {
        "collection": {"stock_name": " 삼성전자 ", "stock_code": "5930",
                       "start_date": "2026-11-01", "end_date": "2026-11-05",
                       "collected_at": "2026-11-05T18:00:00+09:00"},
        "items": [{"title": "<b>삼성전자</b> 실적 전망", "description": "<b>매출</b> 전망 요약",
                   "originallink": "https://WWW.HANKYUNG.COM/article/1?utm_source=naver&id=1#section",
                   "pubDate": "Thu, 05 Nov 2026 10:00:00 +0900",
                   "original_article": {"status": "success", "publisher": "한국 경제",
                                        "body": "<p>삼성전자 매출은 10조원이다.</p><p>영업이익은 감소하지 않았다.</p>"}}],
    }


# 정제·메타데이터 표준화·출처·원문 기반 중요도와 raw 불변성 검증
def test_preprocess_news_normalizes_without_mutating_raw(raw_news):
    before = deepcopy(raw_news)
    records = preprocessor.preprocess_news(raw_news)
    assert raw_news == before
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ProcessedNews)
    assert tuple(asdict(record)) == (
        "news_id", "stock_name", "stock_code", "url", "title", "content", "important", "publisher", "published_date",
    )
    assert record.news_id == "261106123045_01"
    assert record.stock_name == "삼성전자"
    assert record.stock_code == "005930"
    assert record.publisher == "한국경제"
    assert record.url == "https://www.hankyung.com/article/1?id=1"
    assert record.title == "삼성전자 실적 전망"
    assert record.content == "삼성전자 매출은 10조원이다.\n영업이익은 감소하지 않았다."
    assert "sentiment" not in asdict(record)
    assert record.important == "O"
    assert record.published_date == "2026-11-05"


# 광고 전용 행·스크립트·장식만 제거하고 금융 수치·부호·기업명·부정 표현 보존
def test_text_cleaner_preserves_financial_meaning():
    content = clean_news_text(
        "<p>SK하이닉스 영업이익 -1.2%, 매출 10,000억원, 환율 ₩1,350/$1이다.</p>"
        "<p>전망은 나쁘지 않으며 PER &lt; 10, 성장률은 +3.5%이다.</p>"
        "<script>광고 추적</script><style>숨김</style>"
        "<p>광고: 지금 가입</p><p>무단 전재 및 재배포 금지</p><p>■ 정상\u200b 본문</p>"
    )
    assert "-1.2%" in content and "10,000억원" in content
    assert "₩1,350/$1" in content and "+3.5%" in content and "PER < 10" in content
    assert "나쁘지 않으며" in content and "SK하이닉스" in content
    assert "광고" not in content and "추적" not in content and "무단" not in content
    assert "■" not in content and "\u200b" not in content
    assert clean_news_text(None) == ""


# 원문 미확보 기사는 문서 기준에 따라 API 요약문을 사용하며 낮은 우선순위 부여
@pytest.mark.parametrize("status", ["unsupported_publisher", "outside_period", "fetch_failed", "empty_body"])
def test_description_fallback_is_explicit(raw_news, status):
    item = raw_news["items"][0]
    item["original_article"] = {"status": status, "body": None}
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.content == "매출 전망 요약"
    assert record.important == "X"
    assert "sentiment" not in asdict(record)


# 제목 또는 모든 분석 텍스트가 없고 날짜·URL이 잘못된 기사는 제거
@pytest.mark.parametrize("failure", ["title", "content", "date", "date_type", "url", "future"])
def test_incomplete_article_is_removed(raw_news, failure):
    item = raw_news["items"][0]
    if failure == "title":
        item["title"] = "<b> </b>"
    elif failure == "content":
        item["original_article"]["body"] = None
        item["description"] = " "
    elif failure == "date":
        item["pubDate"] = "invalid"
    elif failure == "date_type":
        item["pubDate"] = 123
    elif failure == "url":
        item["originallink"] = "javascript:alert(1)"
    else:
        item["pubDate"] = "Thu, 05 Nov 2026 19:00:00 +0900"
    assert preprocessor.preprocess_news(raw_news) == []


# API 요약이 없어도 원문의 모든 문단을 보존
def test_missing_description_preserves_entire_body(raw_news):
    raw_news["items"][0]["description"] = None
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.content == "삼성전자 매출은 10조원이다.\n영업이익은 감소하지 않았다."


# 짧은 API 요약이 있어도 긴 원문 전체를 축약하거나 첫 문단만 저장하지 않음
def test_long_body_is_not_truncated(raw_news):
    body = "\n".join(f"{index}번째 문단. " + "기업 실적과 산업 동향을 확인했다. " * 30 for index in range(8))
    raw_news["items"][0]["original_article"]["body"] = body
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.content == clean_news_text(body)
    assert len(record.content) > 3000
    assert "7번째 문단." in record.content


# UTC 날짜는 한국 날짜로 변환하고 기간 안의 주말 뉴스도 중요도 대상에 포함
def test_kst_date_and_weekend_in_period(raw_news):
    raw_news["items"][0]["pubDate"] = "Sat, 31 Oct 2026 18:00:00 +0000"
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.published_date == "2026-11-01"
    assert record.news_id == "261106123045_01"
    assert record.important == "O"


# 기간 밖 또는 비경제/금융 내용이어도 원문이 있으면 중요도 O
@pytest.mark.parametrize("outside", [False, True])
def test_importance_only_requires_original_body(raw_news, outside):
    if outside:
        raw_news["items"][0]["pubDate"] = "Sat, 31 Oct 2026 10:00:00 +0900"
    else:
        raw_news["items"][0].update(title="삼성전자 행사 사진", description="행사 안내")
        raw_news["items"][0]["original_article"]["body"] = "행사에서 참석자들이 사진을 촬영했다."
    assert preprocessor.preprocess_news(raw_news)[0].important == "O"


# 조회 상태는 중요도에 영향을 주지 않고 실제 본문 값만 확인
def test_body_presence_is_independent_of_fetch_status(raw_news):
    raw_news["items"][0]["original_article"]["status"] = "fetch_failed"
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.important == "O"


# 원문 값이 없거나 공백뿐이면 요약문 대체 및 중요도 X
@pytest.mark.parametrize("body", [None, "", "   "])
def test_missing_original_body_has_low_importance(raw_news, body):
    raw_news["items"][0]["original_article"]["body"] = body
    record = preprocessor.preprocess_news(raw_news)[0]
    assert record.important == "X"
    assert record.content == "매출 전망 요약"


# 추적 URL 차이는 한 기사로 통합하며 본문 확보 기사를 대표로 선택
def test_url_duplicates_prefer_full_article(raw_news):
    duplicate = deepcopy(raw_news["items"][0])
    duplicate["originallink"] = "https://www.hankyung.com/article/1?id=1&fbclid=abc"
    duplicate["original_article"] = {"status": "fetch_failed", "body": None}
    raw_news["items"].insert(0, duplicate)
    records = preprocessor.preprocess_news(raw_news)
    assert len(records) == 1
    assert records[0].important == "O"


# 동일 날짜·제목·내용의 다른 URL 기사는 유사 기사로 통합
def test_similar_articles_are_deduplicated(raw_news):
    duplicate = deepcopy(raw_news["items"][0])
    duplicate["originallink"] = "https://www.mk.co.kr/news/2"
    duplicate["title"] = "삼성전자 실적 전망!"
    raw_news["items"].append(duplicate)
    assert len(preprocessor.preprocess_news(raw_news)) == 1


# 다른 날짜·금융 수치·부정 표현의 후속 보도는 유사해도 제거하지 않음
@pytest.mark.parametrize("difference", ["date", "numbers", "negation"])
def test_meaningful_changes_are_not_deduplicated(raw_news, difference):
    duplicate = deepcopy(raw_news["items"][0])
    duplicate["originallink"] = "https://www.mk.co.kr/news/2"
    if difference == "date":
        duplicate["pubDate"] = "Wed, 04 Nov 2026 10:00:00 +0900"
    elif difference == "numbers":
        duplicate["original_article"]["body"] = duplicate["original_article"]["body"].replace("10조원", "20조원")
    else:
        duplicate["original_article"]["body"] = duplicate["original_article"]["body"].replace("감소하지 않았다", "감소했다")
    raw_news["items"].append(duplicate)
    assert len(preprocessor.preprocess_news(raw_news)) == 2


# 중요도 우선, 같은 중요도에서는 발행일 최신순 정렬
def test_important_news_is_sorted_first(raw_news):
    first = deepcopy(raw_news["items"][0])
    first["originallink"] = "https://www.hankyung.com/article/2"
    first["pubDate"] = "Wed, 04 Nov 2026 10:00:00 +0900"
    fallback = deepcopy(raw_news["items"][0])
    fallback["originallink"] = "https://www.hankyung.com/article/3"
    fallback["original_article"]["body"] = None
    raw_news["items"] = [fallback, first]
    records = preprocessor.preprocess_news(raw_news)
    assert [record.important for record in records] == ["O", "X"]


# 파일 전처리는 raw를 보존하며 같은 파일명의 processed JSON에만 기록
def test_preprocess_and_save_snapshot(tmp_path, raw_news, monkeypatch):
    raw_path = tmp_path / "raw" / "005930_news_20261105.json"
    raw_path.parent.mkdir()
    raw_path.write_text(json.dumps(raw_news, ensure_ascii=False), encoding="utf-8")
    before = raw_path.read_bytes()
    destination = tmp_path / "processed" / raw_path.name
    monkeypatch.setattr(preprocessor, "get_processed_news_path", lambda path: destination)
    records = preprocessor.preprocess_and_save_news(raw_path)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved == [asdict(record) for record in records]
    assert all(isinstance(record, ProcessedNews) for record in records)
    assert all("content" in record and "summary" not in record and "content_source" not in record for record in saved)
    assert saved[0]["content"] == "삼성전자 매출은 10조원이다.\n영업이익은 감소하지 않았다."
    assert raw_path.read_bytes() == before
    with pytest.raises(ValueError, match="must not overwrite raw"):
        preprocessor.preprocess_and_save_news(raw_path, raw_path)


# 손상된 raw 구조는 출력 파일을 덮어쓰지 않음
@pytest.mark.parametrize("failure", ["metadata", "stock_code", "period", "items"])
def test_invalid_raw_preserves_output(tmp_path, raw_news, failure):
    if failure == "metadata":
        raw_news.pop("collection")
    elif failure == "stock_code":
        raw_news["collection"]["stock_code"] = "../invalid"
    elif failure == "period":
        raw_news["collection"]["start_date"] = "2026-12-01"
    else:
        raw_news["items"] = [None]
    raw_path, processed_path = tmp_path / "raw.json", tmp_path / "processed.json"
    raw_path.write_text(json.dumps(raw_news), encoding="utf-8")
    processed_path.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        preprocessor.preprocess_and_save_news(raw_path, processed_path)
    assert processed_path.read_text(encoding="utf-8") == "existing"


# 발행 시각과 관계없이 현재 전처리 시각과 결과 순서대로 ID 부여
def test_news_ids_use_current_time_and_result_order(raw_news):
    second = deepcopy(raw_news["items"][0])
    second["originallink"] = "https://www.hankyung.com/article/2"
    second["pubDate"] = "Wed, 04 Nov 2026 10:00:30 +0900"
    second["original_article"]["body"] = "매출은 20조원이다."
    raw_news["items"].append(second)
    records = preprocessor.preprocess_news(raw_news)
    assert [record.news_id for record in records] == ["261106123045_01", "261106123045_02"]


# 중복 제거된 기사는 순번에서 제외
def test_duplicate_news_does_not_consume_an_id(raw_news):
    raw_news["items"].append(deepcopy(raw_news["items"][0]))
    records = preprocessor.preprocess_news(raw_news)
    assert len(records) == 1
    assert records[0].news_id == "261106123045_01"


# 같은 URL을 다시 전처리해도 이전 ID를 조회하지 않고 현재 시각으로 새로 부여
def test_reprocessing_uses_new_current_time_without_extra_files(tmp_path, raw_news):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw_news), encoding="utf-8")
    first = preprocessor.preprocess_and_save_news(raw_path, tmp_path / "first.json")[0]
    repeated = preprocessor.preprocess_and_save_news(
        raw_path, tmp_path / "second.json", now=datetime.fromisoformat("2026-11-07T15:00:00+09:00"),
    )[0]
    assert first.news_id == "261106123045_01"
    assert repeated.news_id == "261107150000_01"
    assert {path.name for path in tmp_path.iterdir()} == {"raw.json", "first.json", "second.json"}


# 한 번에 99건을 넘어도 순번을 잘라 중복시키지 않음
def test_news_id_sequence_can_exceed_two_digits(raw_news):
    template = raw_news["items"][0]
    raw_news["items"] = []
    for index in range(101):
        item = deepcopy(template)
        item["originallink"] = f"https://www.hankyung.com/article/{index}"
        item["original_article"]["body"] = f"매출은 {index}조원이다."
        raw_news["items"].append(item)
    ids = {record.news_id for record in preprocessor.preprocess_news(raw_news)}
    assert len(ids) == 101
    assert "261106123045_100" in ids


# UTC 시각을 한국 시각으로 변환하며 실행마다 순번을 01부터 시작
def test_current_time_is_converted_to_kst(raw_news):
    first = preprocessor.preprocess_news(
        raw_news, now=datetime.fromisoformat("2026-11-06T03:30:45+00:00"),
    )[0]
    second = preprocessor.preprocess_news(
        raw_news, now=datetime.fromisoformat("2026-11-06T03:30:46+00:00"),
    )[0]
    assert first.news_id == "261106123045_01"
    assert second.news_id == "261106123046_01"


# 타임존이 없는 전처리 시각은 시스템 시간대에 의존하지 않고 거부
def test_preprocessing_clock_requires_timezone(raw_news):
    with pytest.raises(ValueError, match="timezone"):
        preprocessor.preprocess_news(raw_news, now=datetime(2026, 11, 6, 12, 30, 45))

import pytest
from unittest.mock import Mock

from src.parsers import stock_question_parser as parser
from src.parsers.stock_question_parser import parse_stock_question


# 로컬 매핑 파일과 외부 API에 의존하지 않는 종목 기준정보 제공
@pytest.fixture(autouse=True)
def references(monkeypatch):
    records = [
        {"stock_name": "삼성전자", "stock_code": "005930", "market": "KOSPI"},
        {"stock_name": "하이브", "stock_code": "352820", "market": "KOSPI"},
        {"stock_name": "SK하이닉스", "stock_code": "000660", "market": "KOSPI"},
    ]
    loader = Mock(return_value=records)
    collector = Mock(return_value=records)
    monkeypatch.setattr(parser, "load_stock_references", loader)
    monkeypatch.setattr(parser, "refresh_stock_references", collector)
    return loader, collector


def test_parse_stock_question_extracts_buy_opinion():
    parsed_question = parse_stock_question("삼성전자 사도 될까?")

    assert parsed_question.stock_name == "삼성전자"
    assert parsed_question.stock_code == "005930"
    assert parsed_question.intent == "buy_opinion"


def test_parse_stock_question_extracts_decline_reason():
    parsed_question = parse_stock_question("하이브 하락하는 이유 뭐야?")

    assert parsed_question.stock_name == "하이브"
    assert parsed_question.stock_code == "352820"
    assert parsed_question.intent == "decline_reason"


def test_parse_stock_question_returns_general_analysis():
    parsed_question = parse_stock_question("SK하이닉스 전망 어때?")

    assert parsed_question.stock_name == "SK하이닉스"
    assert parsed_question.stock_code == "000660"
    assert parsed_question.intent == "general_analysis"


def test_parse_stock_question_extracts_stock_code():
    parsed_question = parse_stock_question("005930 사도 될까?")

    assert parsed_question.stock_name == "삼성전자"
    assert parsed_question.stock_code == "005930"
    assert parsed_question.intent == "buy_opinion"


def test_parse_stock_question_raises_for_empty_question():
    with pytest.raises(ValueError, match="Question must not be empty"):
        parse_stock_question(" ")


def test_parse_stock_question_raises_for_unknown_stock():
    with pytest.raises(ValueError, match="Stock name or stock code not found"):
        parse_stock_question("카카오 사도 될까?")


# 기존 파일에 매칭되는 종목이 있으면 수집을 호출하지 않음
def test_existing_stock_does_not_refresh(references):
    _, collector = references
    parse_stock_question("삼성전자 전망")
    collector.assert_not_called()


# 파일이 없을 때 최초 수집 후 바로 질문 분석
def test_missing_file_collects_once(references):
    loader, collector = references
    loader.side_effect = FileNotFoundError
    assert parse_stock_question("삼성전자 전망").stock_code == "005930"
    collector.assert_called_once_with()


# 매칭 실패 시 갱신된 목록에서 신규 종목 재검색
def test_refresh_finds_new_stock(references):
    _, collector = references
    collector.return_value = [
        {"stock_name": "카카오", "stock_code": "035720", "market": "KOSPI"},
    ]
    assert parse_stock_question("카카오 사도 될까?").stock_code == "035720"
    collector.assert_called_once_with()


# 최초 수집에서도 종목을 찾지 못하면 중복 갱신 없이 오류
def test_missing_file_unknown_stock_refreshes_only_once(references):
    loader, collector = references
    loader.side_effect = FileNotFoundError
    with pytest.raises(ValueError, match="Stock name or stock code not found"):
        parse_stock_question("없는종목 전망")
    collector.assert_called_once_with()


# 우선주 이름이 보통주 이름에 포함되어도 긴 이름을 우선 매칭
def test_longer_stock_name_matches_first(references):
    loader, _ = references
    loader.return_value = loader.return_value + [
        {"stock_name": "삼성전자우", "stock_code": "005935", "market": "KOSPI"},
    ]
    assert parse_stock_question("삼성전자우 전망").stock_code == "005935"


# 질문에 지정된 거래일 수와 기본값 검증
@pytest.mark.parametrize("question,expected", [
    ("삼성전자 전망", 5), ("삼성전자 최근 10거래일 전망", 10),
    ("삼성전자 최근 3일 흐름", 3), ("005930 최근 1 거래일", 1),
])
def test_parse_recent_days(question, expected):
    parsed = parse_stock_question(question)
    assert parsed.recent_days == expected
    assert parsed.to_dict()["recent_days"] == expected


# 전일 종가용 1봉을 포함한 단일 호출 한도를 넘는 기간은 오류
@pytest.mark.parametrize("days", [0, 200])
def test_invalid_recent_days(days):
    with pytest.raises(ValueError, match="recent_days"):
        parse_stock_question(f"삼성전자 최근 {days}거래일")

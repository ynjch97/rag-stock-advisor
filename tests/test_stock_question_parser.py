import pytest

from src.parsers.stock_question_parser import parse_stock_question


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

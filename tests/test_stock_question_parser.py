import pytest
from datetime import date
from unittest.mock import Mock

from src.parsers import stock_question_parser as parser
from src.parsers.stock_question_parser import parse_stock_question
from src.parsers.llm_question_analyzer import QuestionAnalysis


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
    cases = {
        "삼성전자 사도 될까?": ("삼성전자", "삼성전자", "buy_opinion", None),
        "하이브 하락하는 이유 뭐야?": ("하이브", "하이브", "decline_reason", None),
        "SK하이닉스 전망 어때?": ("SK하이닉스", "SK하이닉스", "general_analysis", None),
        "005930 사도 될까?": ("005930", None, "buy_opinion", None),
        "카카오 사도 될까?": ("카카오", "카카오", "buy_opinion", None),
        "삼성전자 전망": ("삼성전자", "삼성전자", "general_analysis", None),
        "없는종목 전망": ("없는종목", "없는종목", "general_analysis", None),
        "삼성전자우 전망": ("삼성전자우", "삼성전자우", "general_analysis", None),
        "삼성전자 최근 10거래일 전망": ("삼성전자", "삼성전자", "general_analysis", 10),
        "삼성전자 최근 3거래일 흐름": ("삼성전자", "삼성전자", "general_analysis", 3),
        "005930 최근 1 거래일": ("005930", None, "general_analysis", 1),
        "삼성전자 최근 0거래일": ("삼성전자", "삼성전자", "general_analysis", 0),
        "삼성전자 최근 200거래일": ("삼성전자", "삼성전자", "general_analysis", 200),
    }

    # 질문 해석은 미리 지정한 모의 LLM 결과로 주입; 규칙 기반 대체 분석은 사용하지 않음
    def analyze(question, *args):
        mention, name, intent, days = cases[question]
        return QuestionAnalysis.model_validate({
            "discovery": None,
            "stocks": [{"mention": mention, "stock_name": name, "role": "subject"}],
            "intent": intent, "period": {"kind": "default" if days is None else "trading_days",
                "recent_days": days, "start_date": None, "end_date": None},
            "needs_clarification": False, "clarification_question": None,
        })

    monkeypatch.setattr(parser, "analyze_stock_question", analyze)
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
    assert parse_stock_question("카카오 사도 될까?").needs_clarification


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
    assert parse_stock_question("없는종목 전망").needs_clarification
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
    ("삼성전자 최근 3거래일 흐름", 3), ("005930 최근 1 거래일", 1),
])
def test_parse_recent_days(question, expected):
    parsed = parse_stock_question(question)
    assert parsed.recent_days == expected
    assert parsed.to_dict()["period"]["recent_days"] == expected


# 전일 종가용 1봉을 포함한 단일 호출 한도를 넘는 기간은 오류
@pytest.mark.parametrize("days", [0, 200])
def test_invalid_recent_days(days):
    with pytest.raises(ValueError, match="recent_days"):
        parse_stock_question(f"삼성전자 최근 {days}거래일")


# 별칭 두 개를 LLM으로 해석한 뒤 기준정보 코드와 비교 역할 확정
def test_llm_aliases_and_multiple_stocks(monkeypatch, references):
    analysis = QuestionAnalysis.model_validate({
        "discovery": None,
        "stocks": [
            {"mention": "삼전", "stock_name": "삼성전자", "role": "target"},
            {"mention": "하이닉스", "stock_name": "SK하이닉스", "role": "comparison"},
        ],
        "intent": "comparison_reason",
        "period": {"kind": "trading_days", "recent_days": 10, "start_date": None, "end_date": None},
        "needs_clarification": False, "clarification_question": None,
    })
    analyzer = Mock(return_value=analysis)
    monkeypatch.setattr(parser, "analyze_stock_question", analyzer)
    parsed = parse_stock_question("삼전이 하이닉스보다 최근 10거래일 오르는 이유는?")
    assert [stock.stock_code for stock in parsed.stocks] == ["005930", "000660"]
    assert [stock.role for stock in parsed.stocks] == ["target", "comparison"]
    assert parsed.recent_days == 10
    assert not parsed.needs_clarification
    analyzer.assert_called_once()


# 달력 기간을 보존하고 현재 조회용 거래일 수로 임의 변환하지 않음
def test_calendar_period_is_preserved(monkeypatch):
    analysis = QuestionAnalysis.model_validate({
        "discovery": None,
        "stocks": [{"mention": "삼성전자", "stock_name": "삼성전자", "role": "subject"}],
        "intent": "general_analysis",
        "period": {"kind": "date_range", "recent_days": None, "start_date": "2026-09-26", "end_date": "2026-10-26"},
        "needs_clarification": False, "clarification_question": None,
    })
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("삼성전자 최근 한 달 전망", reference_date=date(2026, 10, 26))
    assert parsed.period.start_date == "2026-09-26"
    assert parsed.period.recent_days is None
    with pytest.raises(ValueError, match="date range"):
        _ = parsed.recent_days


# 질문에 없는 원문 표현을 모델이 만들어내면 조회 차단
def test_hallucinated_mention_is_rejected(monkeypatch):
    analysis = QuestionAnalysis.model_validate({
        "discovery": None,
        "stocks": [{"mention": "하이닉스", "stock_name": "SK하이닉스", "role": "subject"}],
        "intent": "general_analysis",
        "period": {"kind": "default", "recent_days": None, "start_date": None, "end_date": None},
        "needs_clarification": False, "clarification_question": None,
    })
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    with pytest.raises(ValueError, match="원문 표현"):
        parse_stock_question("삼전 전망")


# 정상 응답·확인 질문 모두 실제 호출 횟수를 직렬화 결과에 포함
@pytest.mark.parametrize("case", ["normal", "clarification", "unknown", "comparison"])
def test_openai_attempts_in_parsed_output(monkeypatch, case):
    analysis = QuestionAnalysis.model_validate({
        "discovery": None,
        "stocks": [{"mention": "삼전", "stock_name": "삼성전자", "role": "subject"}],
        "intent": "comparison_reason" if case == "comparison" else "general_analysis",
        "period": {"kind": "default", "recent_days": None, "start_date": None, "end_date": None},
        "needs_clarification": case == "clarification", "clarification_question": None,
    })
    analysis._openai_call_attempts = 2
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("삼전 전망", stock_infos=[] if case == "unknown" else None)
    assert parsed.openai_call_attempts == 2
    assert parsed.to_dict()["openai_call_attempts"] == 2
    assert parsed.needs_clarification == (case != "normal")


# 후보를 만들지 않고 발굴 조건만 내놓는 LLM 분석 대역
def _discovery_analysis(discovery, **options):
    payload = {
        "stocks": [], "intent": "stock_discovery", "discovery": discovery,
        "period": {"kind": "default", "recent_days": None, "start_date": None, "end_date": None},
        "needs_clarification": False, "clarification_question": None,
    }
    return QuestionAnalysis.model_validate({**payload, **options})


# 종목이 없어도 테마·요청 개수로 정상 파싱하며 기준정보·외부 수집을 호출하지 않음
def test_discovery_without_specified_stock(monkeypatch, references):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 3})
    analysis._openai_call_attempts = 2
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("반도체 소재 관련주 종목 3가지만 추천해줘")
    assert parsed.intent == "stock_discovery"
    assert parsed.stocks == () and not parsed.needs_clarification
    assert parsed.to_dict()["discovery"] == {"theme": "반도체 소재", "requested_count": 3}
    assert parsed.recent_days == 5
    assert parsed.openai_call_attempts == 2
    references[0].assert_not_called()
    references[1].assert_not_called()
    with pytest.raises(ValueError, match="discovery"):
        _ = parsed.stock_code


# 개수 미지정은 기본 3개이며 작은 명시 개수는 유지
@pytest.mark.parametrize("count,expected", [(None, 3), (1, 1), (2, 2), (3, 3), (5, 5), (10, 10)])
def test_discovery_requested_count(monkeypatch, count, expected):
    analysis = _discovery_analysis({"theme": " 반도체 소재 ", "requested_count": count})
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("반도체 소재 관련주 추천")
    assert parsed.discovery.theme == "반도체 소재"
    assert parsed.discovery.requested_count == expected
    assert not parsed.needs_clarification


# 상한 초과 요청은 잘라내지 않고 원래 개수를 보존하여 확인 요청
def test_discovery_count_over_limit_requires_clarification(monkeypatch, references):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 11})
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("반도체 소재 관련주 11개 추천")
    assert parsed.needs_clarification
    assert parsed.discovery.requested_count == 11
    assert "최대 10개" in parsed.clarification_question
    references[0].assert_not_called()
    references[1].assert_not_called()


# 테마가 없으면 발굴 의도를 유지하면서 조건 확인만 요청
@pytest.mark.parametrize("criteria", [None, {"theme": None, "requested_count": 3},
                                      {"theme": " ", "requested_count": 3}])
def test_discovery_missing_theme(monkeypatch, references, criteria):
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=_discovery_analysis(criteria)))
    parsed = parse_stock_question("관련주 3개 추천")
    assert parsed.needs_clarification and parsed.intent == "stock_discovery"
    assert parsed.discovery is None
    assert "업종·테마" in parsed.clarification_question
    references[0].assert_not_called()


# LLM의 확인 필요 판단과 추출 가능한 조건은 함께 보존
def test_discovery_preserves_llm_clarification(monkeypatch):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 3},
                                   needs_clarification=True, clarification_question="어떤 소재 분야인가요?")
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("반도체 소재 관련주 추천")
    assert parsed.clarification_question == "어떤 소재 분야인가요?"
    assert parsed.discovery.theme == "반도체 소재"


# 발굴 질문에 실제 등장한 기준 종목은 기존 별칭·코드 검증을 그대로 적용
def test_discovery_with_reference_stock(monkeypatch, references):
    analysis = _discovery_analysis({"theme": "삼성전자 관련 반도체 소재", "requested_count": 2},
                                   stocks=[{"mention": "삼전", "stock_name": "삼성전자", "role": "subject"}])
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("삼전 관련 반도체 소재주 2개 추천")
    assert parsed.stock_code == "005930" and parsed.discovery.requested_count == 2
    assert not parsed.needs_clarification
    references[0].assert_called_once()
    references[1].assert_not_called()


# 추천 후보를 질문 속 종목으로 위장한 출력은 기준정보 조회 전에 차단
def test_discovery_cannot_inject_candidate_stocks(monkeypatch, references):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 3},
                                   stocks=[{"mention": "하이닉스", "stock_name": "SK하이닉스", "role": "subject"}])
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    with pytest.raises(ValueError, match="원문 표현"):
        parse_stock_question("반도체 소재 관련주 3개 추천")
    references[0].assert_not_called()
    references[1].assert_not_called()


# 발굴 개수와 기간은 독립 필드이며 달력 기간을 임의의 거래일 수로 바꾸지 않음
def test_discovery_preserves_explicit_period(monkeypatch):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 3}, period={
        "kind": "date_range", "recent_days": None, "start_date": "2026-10-01", "end_date": "2026-10-31",
    })
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    parsed = parse_stock_question("지난달 반도체 소재 관련주 3개 추천", reference_date=date(2026, 11, 13))
    assert parsed.period.start_date == "2026-10-01"
    assert parsed.discovery.requested_count == 3
    with pytest.raises(ValueError, match="date range"):
        _ = parsed.recent_days


# 일반 분석에 발굴 조건을 섞은 불일치 출력 거부
def test_non_discovery_intent_rejects_discovery_criteria(monkeypatch):
    analysis = _discovery_analysis({"theme": "반도체 소재", "requested_count": 3}, intent="general_analysis")
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    with pytest.raises(ValueError, match="Only stock_discovery"):
        parse_stock_question("삼성전자 전망")


# 기존 지정 종목 분석은 여전히 종목 누락을 확인 질문으로 처리
def test_missing_stock_in_analysis_still_requires_clarification(monkeypatch):
    analysis = _discovery_analysis(None, intent="general_analysis")
    monkeypatch.setattr(parser, "analyze_stock_question", Mock(return_value=analysis))
    assert parse_stock_question("주가 전망").needs_clarification

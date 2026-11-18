from collections import Counter
import json
from unittest.mock import patch

from src.app.stock_advice_workflow import run_stock_advice_workflow
from src.parsers.stock_question_parser import parse_stock_question
from src.services import stock_discovery_service as s


QUESTION = "전기차 관련주 4개만 추천해줘"


# 키워드로 검색된 뉴스 조회 (실제 API 비용 발생)
def inspect_discovery_stages():
    fetch = s.collect_theme_news
    analyze = s.analyze_discovery_candidates

    # 실제 뉴스 수집 결과의 검색어·기사 수·원문 상태를 출력
    def inspect_news(*args, **kwargs):
        payload = fetch(*args, **kwargs)
        print("실제 검색어:", args[0])
        print("검색 건수:", len(payload["items"]))
        print("원문 상태:", dict(Counter(
            item.get("original_article", {}).get("status") for item in payload["items"]
        )))
        return payload

    # LLM 입력 원문 수와 검증 전 추출 후보를 출력
    def inspect_analysis(*args, **kwargs):
        print("분석 가능한 원문:", len(args[1]))
        result = analyze(*args, **kwargs)
        print("LLM 추출 후보:", [
            (candidate.stock_name, [evidence.relation for evidence in candidate.evidence])
            for candidate in result.candidates
        ])
        return result

    parsed = parse_stock_question(QUESTION)
    print("분석 조건:", parsed.discovery)
    if parsed.needs_clarification or parsed.discovery is None:
        print("추가 확인 필요:", parsed.clarification_question)
        return
    with patch.object(s, "collect_theme_news", side_effect=inspect_news), \
            patch.object(s, "analyze_discovery_candidates", side_effect=inspect_analysis):
        result = s.discover_stocks(parsed.discovery, start=1)
    print("검증 통과 후보:", [(candidate.stock_name, candidate.stock_code) for candidate in result["candidates"]])
    print("페이지 정보:", result["collection"])


# 질문 분석 결과인 ParsedQuestion 조회 + 검색된 후보 목록 조회 (실제 API 비용 발생)
def inspect_discovery_candidates():
    parsed = parse_stock_question(QUESTION)
    print("[ParsedQuestion]")
    print(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2))

    result = (
        s.discover_stocks(parsed.discovery, start=1)
        if not parsed.needs_clarification and parsed.discovery is not None
        else {"candidates": []}
    )
    print("[후보 목록]")
    print([(candidate.stock_name, candidate.stock_code, candidate.selection_reason) for candidate in result["candidates"]])


# workflow 연결하여 테스트 : Agent 판단 + 추가 조회 포함
def inspect_advice_workflow():
    print(run_stock_advice_workflow(QUESTION))


if __name__ == "__main__":
    # 실행할 함수 하나만 주석을 해제
    # inspect_discovery_stages()      # 키워드로 검색된 뉴스 조회
    inspect_discovery_candidates()  # 질문 분석 결과인 ParsedQuestion 조회 + 검색된 후보 목록 조회
    # inspect_advice_workflow()       # workflow 연결하여 테스트
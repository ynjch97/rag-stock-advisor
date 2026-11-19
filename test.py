import json

from src.app.stock_advice_workflow import run_stock_advice_workflow
from src.parsers.stock_question_parser import parse_stock_question
from src.services import stock_discovery_service as s


QUESTION = "우리로 주가 분석"


# 질문 분석 결과인 ParsedQuestion 조회 + 검색된 후보 목록 조회 (실제 API 비용 발생)
def inspect_discovery_candidates():
    parsed = parse_stock_question(QUESTION)
    print("[ParsedQuestion]")
    print(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2))

    if parsed.needs_clarification or parsed.discovery is None:
        print(parsed.clarification_question or "종목 발굴 조건이 없는 질문입니다.")
        return

    result = s.discover_theme_stocks(parsed.discovery, start=1)
    print("[후보 목록]")
    print(json.dumps([
        {
            "stock_name": candidate.stock_name,
            "stock_code": candidate.stock_code,
            "selection_reason": candidate.selection_reason,
        }
        for candidate in result["candidates"]
    ], ensure_ascii=False, indent=2))


# workflow 연결하여 테스트 : Agent 판단 + 추가 조회 포함
def inspect_advice_workflow():
    print(run_stock_advice_workflow(QUESTION))


if __name__ == "__main__":
    # 실행할 함수 하나만 주석을 해제
    # inspect_discovery_candidates()  # 질문 분석 결과인 ParsedQuestion 조회 + 검색된 후보 목록 조회
    inspect_advice_workflow()       # workflow 연결하여 테스트

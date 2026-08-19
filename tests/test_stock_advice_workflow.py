from src.app.stock_advice_workflow import run_stock_advice_workflow


def test_run_stock_advice_workflow_generates_answer():
    answer = run_stock_advice_workflow("삼성전자 사도 될까?")

    assert "[삼성전자 간단 분석]" in answer
    assert "현재가는" in answer
    assert "뉴스 흐름은" in answer
    assert "투자 참고용" in answer

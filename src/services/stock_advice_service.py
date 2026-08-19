from typing import Any


StockContext = dict[str, Any]


# 분석 컨텍스트를 바탕으로 자연어 투자 참고 응답 생성
def generate_stock_advice(context: StockContext) -> str:
    stock = context["stock"]
    question = context["question"]
    summary = context["summary"]
    news = context["news"]

    stock_overview = _build_stock_overview(stock)
    news_overview = _build_news_overview(summary)
    key_news = _build_key_news(news)
    conclusion = _build_conclusion(question["intent"], stock, summary)

    return "\n".join(
        [
            f"[{stock['stock_name']} 간단 분석]",
            "",
            stock_overview,
            news_overview,
            key_news,
            "",
            conclusion,
            "",
            "※ 이 응답은 투자 참고용이며 최종 투자 판단은 본인 책임입니다.",
        ]
    )


# 주가 흐름 요약 문장 생성
def _build_stock_overview(stock: dict[str, Any]) -> str:
    return (
        f"현재가는 {stock['current_price']:,}원이고, "
        f"등락률은 {stock['change_rate']}%입니다. "
        f"최근 흐름은 {stock['trend']}입니다."
    )


# 뉴스 감성 요약 문장 생성
def _build_news_overview(summary: dict[str, int]) -> str:
    positive_count = summary["positive_news_count"]
    neutral_count = summary["neutral_news_count"]
    negative_count = summary["negative_news_count"]

    return (
        "뉴스 흐름은 "
        f"긍정 {positive_count}건, "
        f"중립 {neutral_count}건, "
        f"부정 {negative_count}건으로 확인됩니다."
    )


# 주요 뉴스 요약 문장 생성
def _build_key_news(news: list[dict[str, Any]]) -> str:
    if not news:
        return "확인된 관련 뉴스는 없습니다."

    news_lines = [
        f"- {item['title']}: {item['summary']}"
        for item in news[:3]
    ]

    return "주요 뉴스:\n" + "\n".join(news_lines)


# 질문 의도와 데이터 흐름을 바탕으로 결론 문장 생성
def _build_conclusion(
    intent: str,
    stock: dict[str, Any],
    summary: dict[str, int],
) -> str:
    positive_count = summary["positive_news_count"]
    negative_count = summary["negative_news_count"]
    change_rate = stock["change_rate"]

    if intent == "buy_opinion":
        return _build_buy_opinion_conclusion(
            change_rate,
            positive_count,
            negative_count,
        )

    if intent == "decline_reason":
        return _build_decline_reason_conclusion(
            change_rate,
            negative_count,
        )

    return "종합하면 현재는 주가 흐름과 뉴스 분위기를 함께 확인하면서 신중하게 판단할 필요가 있습니다."


# 매수 판단 질문에 대한 결론 생성
def _build_buy_opinion_conclusion(
    change_rate: float,
    positive_count: int,
    negative_count: int,
) -> str:
    if change_rate > 0 and positive_count >= negative_count:
        return "종합하면 단기 분위기는 나쁘지 않지만, 추격 매수보다는 가격 변동을 확인하며 분할 접근하는 편이 적절합니다."

    if change_rate < 0 or negative_count > positive_count:
        return "종합하면 지금은 적극 매수보다는 관망에 가깝습니다. 단기 변동성과 부정 요인을 먼저 확인하는 편이 좋습니다."

    return "종합하면 뚜렷한 방향성이 강하지 않아, 추가 뉴스와 주가 흐름을 확인한 뒤 판단하는 편이 적절합니다."


# 하락 이유 질문에 대한 결론 생성
def _build_decline_reason_conclusion(
    change_rate: float,
    negative_count: int,
) -> str:
    if change_rate < 0 and negative_count > 0:
        return "하락 원인은 단기 주가 약세와 부정 뉴스가 함께 작용했을 가능성이 있습니다."

    if change_rate < 0:
        return "하락은 확인되지만, 샘플 뉴스만으로는 명확한 원인을 단정하기 어렵습니다."

    return "현재 샘플 데이터 기준으로는 뚜렷한 하락 흐름이 강하게 확인되지는 않습니다."

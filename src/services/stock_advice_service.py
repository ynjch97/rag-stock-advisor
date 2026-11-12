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
    if context.get("data_sources", {}).get("news") == "sample":
        news_overview = "샘플 " + news_overview
    key_news = _build_key_news(news)
    conclusion = _build_conclusion(question["intent"], stock)

    return "\n".join(
        [
            f"[{stock['stock_name']} 간단 분석]",
            "",
            stock_overview,
            news_overview,
            key_news,
            "",
            conclusion,
            *(["", "데이터 확인 한계:", *[f"- {issue}" for issue in context["issues"]]]
              if context.get("issues") else []),
            "",
            "※ 이 응답은 투자 참고용이며 최종 투자 판단은 본인 책임입니다.",
        ]
    )


# 복수 종목의 분석을 합치고 같은 기준일·기간일 때만 상대 흐름 비교
def generate_stock_comparison_advice(contexts: list[StockContext]) -> str:
    sections = [generate_stock_advice(context) for context in contexts]
    trends = [context["stock_trend"] for context in contexts]
    if all(trends) and len({(trend["base_date"], trend["recent_days"]) for trend in trends}) == 1:
        strongest = max(trends, key=lambda trend: trend["recent_change_rate"])
        weakest = min(trends, key=lambda trend: trend["recent_change_rate"])
        if strongest["recent_change_rate"] == weakest["recent_change_rate"]:
            comparison = "동일 기간의 종가 변화율은 같습니다."
        else:
            comparison = f"동일 기간에는 {strongest['stock_name']}의 가격 흐름이 상대적으로 더 강했습니다."
    else:
        comparison = "동일 기준일·기간의 실제 추세가 확보되지 않아 상대 흐름을 확정하지 않았습니다."
    return "\n\n".join(sections + [comparison, "조회된 뉴스는 원인 후보이며 가격 변동과의 인과관계는 추가 확인이 필요합니다."])


# 주가 흐름 요약 문장 생성
def _build_stock_overview(stock: dict[str, Any]) -> str:
    if stock.get("current_price") is None:
        return "기준일의 주가·추세를 확보하지 못했습니다."
    label = f"{stock['base_date']} 종가는" if stock.get("price_type") == "close" else "현재가는"
    return (
        f"{label} {stock['current_price']:,}원이고, "
        f"최근 흐름은 {stock['trend']}입니다."
    )


# 확인된 뉴스 건수 요약 문장 생성
def _build_news_overview(summary: dict[str, int]) -> str:
    return f"뉴스 흐름은 관련 기사 {summary['news_count']}건으로 확인됩니다."


# 기사별 청크를 묶어 상위 3개 기사의 본문 발췌·출처 표시, 전체 본문은 컨텍스트에 유지
def _build_key_news(news: list[dict[str, Any]]) -> str:
    if not news:
        return "확인된 관련 뉴스는 없습니다."

    articles = {}
    for item in news:
        key = item.get("url") or item.get("news_id") or (item["title"], item["published_date"])
        articles.setdefault(key, []).append(item)
    news_lines = []
    for chunks in list(articles.values())[:3]:
        item = chunks[0]
        content = "\n".join(
            chunk.get("content") or chunk.get("summary", "")
            for chunk in sorted(chunks, key=lambda chunk: chunk.get("chunk_index", 0))
        )
        excerpt = content[:300] + ("..." if len(content) > 300 else "")
        source = f"{item.get('publisher') or '출처 미상'} | {item['published_date']}"
        news_lines.append(f"- {item['title']}: {excerpt}\n  [{source}]"
                          + (f" {item['url']}" if item.get("url") else ""))

    return "주요 뉴스:\n" + "\n".join(news_lines)


# 질문 의도와 데이터 흐름을 바탕으로 결론 문장 생성
def _build_conclusion(
    intent: str,
    stock: dict[str, Any],
) -> str:
    if stock.get("change_rate") is None and intent in ("rise_reason", "decline_reason"):
        return "조회된 주가가 없어 상승·하락 여부를 확인할 수 없습니다."
    if intent == "buy_opinion":
        return "주가 흐름만으로 매수 적정성을 판단할 수 없습니다. 기업 실적과 관련 뉴스의 근거를 추가로 확인해야 합니다."

    if intent == "decline_reason":
        if stock["change_rate"] < 0:
            return "하락은 확인되지만, 조회된 뉴스와의 인과관계는 추가 확인이 필요합니다."
        return "조회된 주가 기준으로는 뚜렷한 하락이 확인되지 않습니다."

    if intent == "rise_reason":
        if stock["change_rate"] > 0:
            return "상승은 확인되지만, 조회된 뉴스와의 인과관계는 추가 확인이 필요합니다."
        return "조회된 주가 기준으로는 뚜렷한 상승이 확인되지 않습니다."

    return "종합하면 주가 흐름과 관련 뉴스의 사실관계를 함께 확인하면서 신중하게 판단할 필요가 있습니다."

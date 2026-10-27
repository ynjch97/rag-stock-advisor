from dataclasses import asdict

from src.collectors.stock_collector import collect_stock_price
from src.config.settings import Settings, settings
from src.data_loaders.sample_news_loader import load_recent_sample_news
from src.data_loaders.sample_stock_loader import load_latest_sample_stock, load_sample_stocks
from src.parsers.stock_question_parser import parse_stock_question
from src.domain.stock_models import DailyStock
from src.domain.advice_models import ParsedQuestion, ParsedStock
from src.preprocessing.stock_preprocessor import load_latest_stock_trend
from src.services.stock_advice_service import generate_stock_advice, generate_stock_comparison_advice
from src.services.stock_context_service import StockContext, build_stock_context


# provider에 따라 샘플 또는 실제 주가·추세를 확보한 뒤 분석 응답 생성
def run_stock_advice_workflow(question: str, app_settings: Settings = settings) -> str:
    provider = (app_settings.stock_api_provider or "").lower()
    if provider not in ("sample", "toss"):
        raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")
    # sample/toss는 주가 제공자만 구분하며 질문 분석은 항상 OpenAI 사용
    stock_infos = None
    if provider == "sample":
        stock_infos = [
            {"stock_name": stock["stock_name"], "stock_code": stock["stock_code"]}
            for stock in load_sample_stocks()
        ]
    parsed_question = parse_stock_question(question, stock_infos=stock_infos, app_settings=app_settings)
    if parsed_question.needs_clarification:
        return parsed_question.clarification_question or "분석할 종목과 기간을 확인해주세요."
    if parsed_question.period.kind == "date_range":
        return (
            f"질문의 기간은 {parsed_question.period.start_date}부터 {parsed_question.period.end_date}까지입니다. "
            "현재 조회는 거래일 수 기준입니다. 최근 몇 거래일을 분석할지 알려주세요."
        )
    contexts = [
        _build_analysis_context(parsed_question, stock, app_settings)
        for stock in parsed_question.stocks
    ]
    if len(contexts) == 1:
        return generate_stock_advice(contexts[0])
    return generate_stock_comparison_advice(contexts)


# 검증된 종목 하나의 가격·추세·뉴스를 기존 컨텍스트 구조로 구성
def _build_analysis_context(
    parsed_question: ParsedQuestion, stock: ParsedStock, app_settings: Settings,
) -> StockContext:
    provider = (app_settings.stock_api_provider or "").lower()
    trend_data = None
    if provider == "sample":
        stock_data = load_latest_sample_stock(stock.stock_name)
        news_data = load_recent_sample_news(stock.stock_name)
    else:
        # 수집 함수 내부에서 완료 일봉 저장과 필요한 추세 전처리까지 수행
        price = collect_stock_price(
            stock.stock_name, stock.stock_code,
            app_settings=app_settings, recent_days=parsed_question.recent_days,
        )
        trend_data = load_latest_stock_trend(
            price.stock_code, parsed_question.recent_days, base_date=price.base_date,
        )
        if isinstance(price, DailyStock) and trend_data.base_date != price.base_date:
            raise ValueError("Daily price and stock trend dates must match.")
        if not isinstance(price, DailyStock) and trend_data.base_date >= price.base_date:
            raise ValueError("Realtime analysis requires a previously completed day's trend.")
        stock_data = asdict(price)
        stock_data.update(
            current_price=price.close if isinstance(price, DailyStock) else price.current_price,
            price_type="close" if isinstance(price, DailyStock) else "realtime",
            source="toss",
        )
        news_data = load_recent_sample_news(stock.stock_name, allow_missing=True)

    return build_stock_context(
        parsed_question=parsed_question,
        stock_data=stock_data,
        news_data=news_data,
        trend_data=trend_data,
    )

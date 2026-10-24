from dataclasses import asdict

from src.collectors.stock_collector import collect_stock_price
from src.config.settings import Settings, settings
from src.data_loaders.sample_news_loader import load_recent_sample_news
from src.data_loaders.sample_stock_loader import load_latest_sample_stock, load_sample_stocks
from src.parsers.stock_question_parser import parse_stock_question
from src.domain.stock_models import DailyStock
from src.preprocessing.stock_preprocessor import load_latest_stock_trend
from src.services.stock_advice_service import generate_stock_advice
from src.services.stock_context_service import build_stock_context


# provider에 따라 샘플 또는 실제 주가·추세를 확보한 뒤 분석 응답 생성
def run_stock_advice_workflow(question: str, app_settings: Settings = settings) -> str:
    provider = (app_settings.stock_api_provider or "").lower()
    if provider not in ("sample", "toss"):
        raise ValueError(f"Unsupported STOCK_API_PROVIDER: {provider or '(not configured)'}")
    # 샘플 모드는 종목 매핑까지 로컬 데이터만 사용하여 API 호출 방지
    stock_infos = None
    if provider == "sample":
        stock_infos = [
            {"stock_name": stock["stock_name"], "stock_code": stock["stock_code"]}
            for stock in load_sample_stocks()
        ]
    parsed_question = parse_stock_question(question, stock_infos=stock_infos)
    trend_data = None
    if provider == "sample":
        stock_data = load_latest_sample_stock(parsed_question.stock_name)
        news_data = load_recent_sample_news(parsed_question.stock_name)
    else:
        # 수집 함수 내부에서 완료 일봉 저장과 필요한 추세 전처리까지 수행
        price = collect_stock_price(
            parsed_question.stock_name, parsed_question.stock_code,
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
        news_data = load_recent_sample_news(parsed_question.stock_name, allow_missing=True)

    context = build_stock_context(
        parsed_question=parsed_question,
        stock_data=stock_data,
        news_data=news_data,
        trend_data=trend_data,
    )

    return generate_stock_advice(context)

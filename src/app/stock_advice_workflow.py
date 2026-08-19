from src.data_loaders.sample_news_loader import load_recent_sample_news
from src.data_loaders.sample_stock_loader import load_latest_sample_stock
from src.parsers.stock_question_parser import parse_stock_question
from src.services.stock_advice_service import generate_stock_advice
from src.services.stock_context_service import build_stock_context


# 사용자 질문부터 자연어 분석 응답까지 MVP 전체 흐름 실행
def run_stock_advice_workflow(question: str) -> str:
    parsed_question = parse_stock_question(question)
    stock_data = load_latest_sample_stock(parsed_question["stock_name"])
    news_data = load_recent_sample_news(parsed_question["stock_name"])

    context = build_stock_context(
        parsed_question=parsed_question,
        stock_data=stock_data,
        news_data=news_data,
    )

    return generate_stock_advice(context)

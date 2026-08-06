import json
from datetime import date
from pathlib import Path
from typing import Any

from src.config.paths import SAMPLE_STOCKS_PATH


StockRecord = dict[str, Any]


# 전체 json을 가져옴
def load_sample_stocks(file_path: Path = SAMPLE_STOCKS_PATH) -> list[StockRecord]:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("sample_stocks.json must contain a JSON array.")

    return data


# 종목명 기준으로 데이터 필터링
def load_sample_stocks_by_name(stock_name: str) -> list[StockRecord]:
    stocks = load_sample_stocks()
    matched_stocks = [
        stock for stock in stocks if stock.get("stock_name") == stock_name
    ]

    if not matched_stocks:
        raise ValueError(f"Sample stock data not found: {stock_name}")

    return matched_stocks


# 종목명 기준으로 가장 최신 데이터 가져옴
def load_latest_sample_stock(stock_name: str) -> StockRecord:
    matched_stocks = load_sample_stocks_by_name(stock_name)

    return max(
        matched_stocks,
        key=lambda stock: date.fromisoformat(stock["base_date"]),
    )

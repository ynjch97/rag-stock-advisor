import pytest

from src.data_loaders.sample_stock_loader import (
    load_latest_sample_stock,
    load_sample_stocks,
    load_sample_stocks_by_name,
)


def test_load_sample_stocks_returns_all_records():
    stocks = load_sample_stocks()

    assert len(stocks) == 32


def test_load_sample_stocks_by_name_filters_stock_name():
    stocks = load_sample_stocks_by_name("삼성전자")

    assert len(stocks) == 8
    assert all(stock["stock_name"] == "삼성전자" for stock in stocks)


def test_load_latest_sample_stock_returns_latest_base_date():
    stock = load_latest_sample_stock("삼성전자")

    assert stock["stock_name"] == "삼성전자"
    assert stock["stock_code"] == "005930"
    assert stock["base_date"] == "2026-07-08"


def test_load_sample_stocks_by_name_raises_for_unknown_stock():
    with pytest.raises(ValueError, match="Sample stock data not found"):
        load_sample_stocks_by_name("카카오")

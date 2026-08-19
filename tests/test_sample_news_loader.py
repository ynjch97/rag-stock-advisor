import pytest

from src.data_loaders.sample_news_loader import (
    load_recent_sample_news,
    load_sample_news,
    load_sample_news_by_name,
)


def test_load_sample_news_returns_all_records():
    news = load_sample_news()

    assert len(news) == 8


def test_load_sample_news_by_name_filters_stock_name():
    news = load_sample_news_by_name("삼성전자")

    assert len(news) == 2
    assert all(item["stock_name"] == "삼성전자" for item in news)


def test_load_recent_sample_news_applies_limit():
    news = load_recent_sample_news("삼성전자", limit=1)

    assert len(news) == 1
    assert news[0]["stock_name"] == "삼성전자"


def test_load_sample_news_by_name_returns_hive_news():
    news = load_sample_news_by_name("하이브")

    assert len(news) == 2
    assert all(item["stock_name"] == "하이브" for item in news)


def test_load_sample_news_by_name_raises_for_unknown_stock():
    with pytest.raises(ValueError, match="Sample news data not found"):
        load_sample_news_by_name("카카오")

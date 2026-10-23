from dataclasses import asdict, replace
import json

import pytest

from src.domain.stock_models import DailyStock
from src.preprocessing.stock_preprocessor import calculate_stock_trend, preprocess_and_save_stock_trend


# 실제 거래일 간격을 가진 일봉과 직전 종가 기반 등락 정보 생성
def _stocks(closes, previous_close=90):
    days = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
    records = []
    for day, close in zip(days, closes):
        records.append(DailyStock(
            stock_name="삼성전자", stock_code="005930", base_date=day,
            open=close, high=close, low=close, close=close, volume=100,
            change_rate=(close - previous_close) / previous_close * 100,
            change_price=close - previous_close,
        ))
        previous_close = close
    return records


# 일봉 5개 수익률과 첫날을 포함한 전일 비교 상승·하락·보합 검증
@pytest.mark.parametrize("closes,previous,rate,up,down,direction", [
    ([100, 110, 105, 105, 120], 90, 20.0, 3, 1, "상승"),
    ([120, 115, 110, 110, 100], 130, -16.67, 0, 4, "하락"),
    ([100, 100, 100, 100, 100], 100, 0.0, 0, 0, "보합"),
])
def test_calculate_stock_trend(closes, previous, rate, up, down, direction):
    trend = calculate_stock_trend(_stocks(closes, previous)[::-1])
    assert trend.recent_days == 5
    assert trend.recent_change_rate == rate
    assert trend.up_days == up
    assert trend.down_days == down
    assert trend.trend == f"최근 5거래일 {direction}"
    assert trend.base_date == "2026-10-07"


# 기간별 최신 n개를 선택하며 1일 수익률은 첫 종가와 마지막 종가가 같아 0
def test_custom_period():
    stocks = _stocks([100, 110, 105, 105, 120])
    assert calculate_stock_trend(stocks, 3).recent_change_rate == 14.29
    assert calculate_stock_trend(stocks, 1).recent_change_rate == 0.0
    assert calculate_stock_trend(stocks, 1).up_days == 1


# 최초 저장·기존 결과 재사용·새 거래일·기간별 이력·원본 정정 검증
def test_save_reuse_and_update_trend(tmp_path):
    path = tmp_path / "prices" / "005930_trend.json"
    stocks = _stocks([100, 110, 105, 105, 120])
    trend, changed = preprocess_and_save_stock_trend(stocks, file_path=path)
    assert changed is True
    assert json.loads(path.read_text(encoding="utf-8")) == [asdict(trend)]
    before = path.stat().st_mtime_ns
    assert preprocess_and_save_stock_trend(stocks, file_path=path) == (trend, False)
    assert path.stat().st_mtime_ns == before

    new_stocks = _stocks([100, 110, 105, 105, 120, 125])
    latest, changed = preprocess_and_save_stock_trend(new_stocks, file_path=path)
    assert changed is True
    assert latest.base_date == "2026-10-08"
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2
    assert preprocess_and_save_stock_trend(new_stocks, 3, path)[1] is True
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 3

    new_stocks[-1] = replace(new_stocks[-1], close=130, change_price=10)
    corrected, changed = preprocess_and_save_stock_trend(new_stocks, file_path=path, data_changed=True)
    assert changed is True
    assert corrected.recent_change_rate != latest.recent_change_rate
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 3
    assert preprocess_and_save_stock_trend(new_stocks, file_path=path, data_changed=True)[1] is False


# 부족한 일봉·중복·혼합 종목·0 종가·잘못된 기간은 저장 전에 거부
@pytest.mark.parametrize("failure", ["short", "duplicate", "mixed", "zero", "period"])
def test_invalid_input_does_not_save(tmp_path, failure):
    stocks = _stocks([100, 110, 105, 105, 120])
    if failure == "short":
        stocks.pop()
    elif failure == "duplicate":
        stocks[-1] = replace(stocks[-1], base_date=stocks[-2].base_date)
    elif failure == "mixed":
        stocks[-1] = replace(stocks[-1], stock_code="000660")
    elif failure == "zero":
        stocks[-1] = replace(stocks[-1], close=0)
    path = tmp_path / "trend.json"
    with pytest.raises(ValueError):
        preprocess_and_save_stock_trend(stocks, 0 if failure == "period" else 5, path)
    assert not path.exists()


# 잘못된 기존 전처리 파일은 조용히 덮어쓰지 않음
def test_invalid_existing_trend_preserved(tmp_path):
    path = tmp_path / "trend.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        preprocess_and_save_stock_trend(_stocks([100] * 5), file_path=path)
    assert path.read_text(encoding="utf-8") == "{}"

from dataclasses import asdict
from datetime import date
from decimal import Decimal
import json
import math
from pathlib import Path
import tempfile

from src.config.paths import get_processed_price_trend_path
from src.domain.stock_models import DEFAULT_RECENT_DAYS, DailyStock, StockTrend


# 완료된 일봉에서 최근 n개를 선택하고 종목·거래일·종가 검증
def _select_recent_stocks(stocks: list[DailyStock], recent_days: int) -> list[DailyStock]:
    if type(recent_days) is not int or not 1 <= recent_days <= 199:
        raise ValueError("recent_days must be between 1 and 199 trading days.")
    if len(stocks) < recent_days:
        raise ValueError(f"At least {recent_days} completed daily stocks are required.")
    selected = sorted(stocks, key=lambda stock: date.fromisoformat(stock.base_date))[-recent_days:]
    if len({stock.base_date for stock in selected}) != recent_days:
        raise ValueError("Duplicate daily stock dates are not allowed.")
    if len({(stock.stock_name, stock.stock_code) for stock in selected}) != 1:
        raise ValueError("Daily stocks must belong to the same stock.")
    if any(type(stock.close) is not int or stock.close <= 0 for stock in selected):
        raise ValueError("Daily stock close prices must be positive integers.")
    if type(selected[0].change_price) is not int or selected[0].close - selected[0].change_price <= 0:
        raise ValueError("The first day's previous close must be positive.")
    return selected


# 첫 종가 대비 마지막 종가 수익률과 n일 각각의 전일 대비 상승·하락 횟수 계산
def calculate_stock_trend(
    stocks: list[DailyStock], recent_days: int = DEFAULT_RECENT_DAYS,
) -> StockTrend:
    selected = _select_recent_stocks(stocks, recent_days)
    rate = (Decimal(selected[-1].close) / Decimal(selected[0].close) - 1) * 100
    differences = [selected[0].change_price] + [
        current.close - previous.close for previous, current in zip(selected, selected[1:])
    ]
    direction = "상승" if rate > 0 else "하락" if rate < 0 else "보합"
    return StockTrend(
        stock_name=selected[-1].stock_name, stock_code=selected[-1].stock_code,
        base_date=selected[-1].base_date, recent_days=recent_days,
        recent_change_rate=float(round(rate, 2)),
        up_days=sum(value > 0 for value in differences),
        down_days=sum(value < 0 for value in differences),
        trend=f"최근 {recent_days}거래일 {direction}",
    )


# 기준일·기간이 같은 추세는 재사용하고 없거나 원본이 변경되면 계산·저장
def preprocess_and_save_stock_trend(
    stocks: list[DailyStock], recent_days: int = DEFAULT_RECENT_DAYS,
    file_path: Path | None = None, *, data_changed: bool = False,
) -> tuple[StockTrend, bool]:
    selected = _select_recent_stocks(stocks, recent_days)
    destination = file_path if file_path is not None else get_processed_price_trend_path(selected[-1].stock_code)
    records = _load_stock_trends(destination, selected[-1].stock_code)
    key = (selected[-1].base_date, recent_days)
    existing = next((record for record in records if (record.base_date, record.recent_days) == key), None)
    if existing is not None and not data_changed:
        return existing, False
    trend = calculate_stock_trend(selected, recent_days)
    if existing == trend:
        return trend, False
    merged = {(record.base_date, record.recent_days): record for record in records}
    merged[key] = trend
    _save_stock_trends(destination, [merged[item] for item in sorted(merged)])
    return trend, True


# 저장된 추세 목록을 읽고 필수 필드·종목·기간·계산값 검증
def _load_stock_trends(file_path: Path, stock_code: str) -> list[StockTrend]:
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        raise ValueError("Stock trends must contain a JSON array.")
    records = []
    for item in data:
        try:
            record = StockTrend(**item)
            valid = (
                record.stock_code == stock_code and isinstance(record.stock_name, str)
                and isinstance(record.trend, str)
                and date.fromisoformat(record.base_date).isoformat() == record.base_date
                and type(record.recent_days) is int and 1 <= record.recent_days <= 199
                and type(record.recent_change_rate) in (float, int) and math.isfinite(record.recent_change_rate)
                and type(record.up_days) is int and type(record.down_days) is int
                and 0 <= record.up_days <= record.recent_days
                and 0 <= record.down_days <= record.recent_days - record.up_days
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Invalid saved stock trend record.")
        records.append(record)
    return records


# 임시 파일로 기록한 뒤 교체하여 기존 추세 이력을 안전하게 저장
def _save_stock_trends(file_path: Path, records: list[StockTrend]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=file_path.parent, suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump([asdict(record) for record in records], file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

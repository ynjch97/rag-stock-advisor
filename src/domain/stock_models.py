from dataclasses import dataclass


@dataclass(frozen=True)
class DailyStock:
    stock_name: str
    stock_code: str
    date: str
    open: int
    high: int
    low: int
    close: int
    volume: int


@dataclass(frozen=True)
class RealtimeStock:
    stock_name: str
    stock_code: str
    current_price: int
    change_rate: float
    base_date: str


@dataclass(frozen=True)
class StockTrend:
    stock_name: str
    stock_code: str
    base_date: str
    recent_days: int
    recent_return_rate: float
    up_days: int
    down_days: int
    trend: str

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

DEFAULT_RECENT_DAYS = 5

@dataclass(frozen=True)
class MarketStatus:
    status: Literal["holiday", "before_market", "regular_market", "after_market"]
    base_date: str
    latest_completed_date: str
    is_realtime: bool
    checked_at: datetime


@dataclass(frozen=True)
class DailyStock:
    stock_name: str
    stock_code: str
    base_date: str
    open: int
    high: int
    low: int
    close: int
    volume: int
    change_rate: float  # 등락률(%)
    change_price: int   # 등락금액


@dataclass(frozen=True)
class RealtimeStock:
    stock_name: str
    stock_code: str
    current_price: int
    change_rate: float  # 등락률(%)
    base_date: str


@dataclass(frozen=True)
class StockTrend:
    stock_name: str
    stock_code: str
    base_date: str
    recent_days: int
    recent_change_rate: float
    up_days: int
    down_days: int
    trend: str

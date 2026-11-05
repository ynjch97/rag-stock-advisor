from dataclasses import dataclass
from typing import Literal

# raw 는 네이버 API 그대로 사용하므로 생략

@dataclass(frozen=True)
class ProcessedNews:
    stock_name: str
    stock_code: str
    url: str
    title: str
    content: str
    important: Literal["O", "X"]
    publisher: str
    published_date: str

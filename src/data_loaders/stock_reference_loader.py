import json
from pathlib import Path

from src.config.paths import STOCK_NAME_CODE_PATH


# 매핑 파일을 읽고 종목명·종목코드·시장 필드와 종목코드 중복 검증
def load_stock_references(
    file_path: Path = STOCK_NAME_CODE_PATH,
) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Stock reference file must contain a non-empty JSON array.")
    codes: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) or not record[field].strip()
            for field in ("stock_name", "stock_code", "market")
        ):
            raise ValueError("Stock reference fields must be non-empty strings.")
        if record["stock_code"] in codes:
            raise ValueError(f"Duplicate stock code: {record['stock_code']}")
        codes.add(record["stock_code"])
    return records

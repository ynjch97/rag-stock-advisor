import json

import pytest

from src.data_loaders.stock_reference_loader import load_stock_references


# 누락 파일의 오류를 최초 수집 판단에 사용할 수 있도록 전달
def test_missing_reference_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stock_references(tmp_path / "missing.json")


# 빈 목록·누락 필드·숫자 종목코드·중복 종목코드 거부 검증
@pytest.mark.parametrize("records", [
    {}, [], [{"stock_name": "삼성전자", "stock_code": 5930, "market": "KOSPI"}],
    [{"stock_name": "삼성전자", "stock_code": "005930"}],
    [{"stock_name": "삼성전자", "stock_code": "005930", "market": "KOSPI"}] * 2,
])
def test_invalid_reference_data(tmp_path, records):
    path = tmp_path / "stock_name_code.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        load_stock_references(path)

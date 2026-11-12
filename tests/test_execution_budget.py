import time

import httpx
import pytest

from src.services.execution_budget import deadline_http_client, remaining_timeout


# 경과한 전체 시간은 다음 요청 시작 전에 차단
def test_expired_budget():
    with pytest.raises(TimeoutError):
        remaining_timeout(time.monotonic() - 1)


# 남은 전체 시간과 요청별 상한 중 작은 값 적용
def test_remaining_timeout():
    assert 0 < remaining_timeout(time.monotonic() + 1, 20) <= 1
    assert remaining_timeout(None, 10) == 10


# 개별 요청의 더 짧은 제한을 늘리지 않고 전체 제한으로도 축소
def test_http_deadline_hook():
    with deadline_http_client(time.monotonic() + 5, 20) as client:
        request = httpx.Request("GET", "https://example.test")
        request.extensions["timeout"] = {key: 1 for key in ("connect", "read", "write", "pool")}
        client.event_hooks["request"][0](request)
        assert all(value == 1 for value in request.extensions["timeout"].values())

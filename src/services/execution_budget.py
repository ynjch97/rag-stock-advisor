import time

import httpx


# 남은 전체 시간과 개별 요청 상한 중 작은 값 반환; 종료 시 새 작업 차단
def remaining_timeout(deadline: float | None, timeout: float = 20.0) -> float:
    remaining = timeout if deadline is None else min(timeout, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("Agent processing time limit reached.")
    return remaining


# 기사 원문 등 연속 HTTP 요청도 매 요청 직전에 남은 시간으로 제한
def deadline_http_client(deadline: float | None, timeout: float = 20.0) -> httpx.Client:
    # httpx가 전송하기 전에 각 타임아웃을 남은 시간 이하로 갱신
    def check_request(request: httpx.Request) -> None:
        limit = remaining_timeout(deadline, timeout)
        original = request.extensions.get("timeout", {})
        request.extensions["timeout"] = {
            key: min(limit, original[key]) if original.get(key) is not None else limit
            for key in ("connect", "read", "write", "pool")
        }

    return httpx.Client(timeout=remaining_timeout(deadline, timeout), event_hooks={"request": [check_request]})

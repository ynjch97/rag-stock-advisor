import math


# 중요도 O 우선, 동일 중요도에서는 가중 합산 점수순으로 최종 청크 선택
def rerank_news_results(results: list[dict], *, top_k: int = 5) -> list[dict]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    for result in results:
        if (not isinstance(result, dict)
                or not isinstance(result.get("chunk_id"), str) or not result["chunk_id"].strip()
                or result.get("important") not in ("O", "X")
                or type(result.get("hybrid_score")) not in (int, float)
                or not math.isfinite(result["hybrid_score"])
                or not 0 <= result["hybrid_score"] <= 1):
            raise ValueError("Reranking requires chunk_id, important, and a hybrid_score between 0 and 1.")
    # 원점수·정규화 점수는 변경하지 않고 동점은 청크 ID로 순서를 고정
    return sorted(results, key=lambda result: (
        result["important"] != "O", -result["hybrid_score"], result["chunk_id"],
    ))[:top_k]

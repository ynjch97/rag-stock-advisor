import math
from pathlib import Path

from kiwipiepy import Kiwi
from openai import OpenAI

from src.config.paths import BM25_TOKENS_PATH, PROJECT_ROOT
from src.config.settings import Settings, settings
from src.search.keyword_search import search_news_keywords
from src.search.search_result_ranker import rerank_news_results
from src.search.vector_search import search_news_vectors


DEFAULT_CANDIDATE_K = 20
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


# 검색별 ID 중복·점수 검증 후 Min-Max 정규화, 동일 점수만 있으면 모두 1
def _normalize_search_results(results: list[dict]) -> tuple[dict[str, dict], dict[str, float]]:
    by_id = {}
    for result in results:
        if (not isinstance(result, dict) or not isinstance(result.get("chunk_id"), str)
                or not result["chunk_id"].strip() or result["chunk_id"] in by_id
                or type(result.get("score")) not in (int, float) or not math.isfinite(result["score"])):
            raise ValueError("Search results must have unique chunk IDs and finite numeric scores.")
        by_id[result["chunk_id"]] = result
    if not by_id:
        return {}, {}
    scores = [result["score"] for result in by_id.values()]
    lowest, highest = min(scores), max(scores)
    normalized = {
        chunk_id: (float((result["score"] - lowest) / (highest - lowest)) if highest != lowest else 1.0)
        for chunk_id, result in by_id.items()
    }
    return by_id, normalized


# 청크 ID 합집합을 만들고 원점수를 보존하며 정규화 점수에 0.7·0.3 가중치 적용
def combine_news_search_results(vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
    vectors, vector_normalized = _normalize_search_results(vector_results)
    keywords, keyword_normalized = _normalize_search_results(keyword_results)
    combined = []
    for chunk_id in sorted(vectors.keys() | keywords.keys()):
        vector, keyword = vectors.get(chunk_id), keywords.get(chunk_id)
        vector_data = {key: value for key, value in vector.items() if key != "score"} if vector is not None else None
        keyword_data = {key: value for key, value in keyword.items() if key != "score"} if keyword is not None else None
        if vector_data is not None and keyword_data is not None and vector_data != keyword_data:
            raise ValueError("Vector and keyword results contain different metadata for the same chunk.")
        # 미등장 쪽 원점수는 None, 가중 합산 기여도는 0; 남은 쪽 가중치를 재조정하지 않음
        vector_score, keyword_score = vector_normalized.get(chunk_id, 0.0), keyword_normalized.get(chunk_id, 0.0)
        combined.append({
            **(vector_data if vector_data is not None else keyword_data),
            "vector_score": float(vector["score"]) if vector is not None else None,
            "keyword_score": float(keyword["score"]) if keyword is not None else None,
            "vector_normalized_score": vector_score,
            "keyword_normalized_score": keyword_score,
            "hybrid_score": VECTOR_WEIGHT * vector_score + KEYWORD_WEIGHT * keyword_score,
        })
    return sorted(combined, key=lambda result: (-result["hybrid_score"], result["chunk_id"]))


# 동일 조건으로 후보를 각 20개 확보하고 가중 합산·중요도 재정렬 후 최종 top_k 반환
def search_news_hybrid(
    question: str, *, top_k: int = 5, candidate_k: int = DEFAULT_CANDIDATE_K,
    stock_code: str | None = None, start_date: str | None = None, end_date: str | None = None,
    faiss_dir: Path | None = None, bm25_path: Path = BM25_TOKENS_PATH,
    app_settings: Settings = settings, client: OpenAI | None = None, kiwi: Kiwi | None = None,
) -> list[dict]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if type(candidate_k) is not int or candidate_k < top_k:
        raise ValueError("candidate_k must be an integer greater than or equal to top_k.")
    directory = faiss_dir if faiss_dir is not None else PROJECT_ROOT / app_settings.vector_db_path
    conditions = dict(top_k=candidate_k, stock_code=stock_code, start_date=start_date,
                      end_date=end_date, app_settings=app_settings)
    # 로컬 BM25 검증을 먼저 수행하여 파일 오류 시 불필요한 임베딩 API 호출 방지
    keyword_results = search_news_keywords(
        question, **conditions, bm25_path=bm25_path,
        metadata_path=directory / "stock_advisor_metadata.json", kiwi=kiwi,
    )
    vector_results = search_news_vectors(question, **conditions, faiss_dir=directory, client=client)
    combined = combine_news_search_results(vector_results, keyword_results)
    return rerank_news_results(combined, top_k=top_k)

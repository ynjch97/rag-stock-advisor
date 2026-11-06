from functools import lru_cache
from importlib.metadata import version
import json
import math
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

# 검색 점수 계산 설정 (기본값으로 우선 세팅함)
  # k1 : 같은 단어가 반복될 때 점수가 증가하는 정도
  # b : 문서 길이 보정 정도 (0이면 길이 무시, 1이면 길이 비례)
  # epsilon : rank_bm25의 기본 보정 방식
BM25_PARAMETERS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}
# 토큰화 방식 기록
  # name: 사용하는 형태소 분석기 이름
  # version: 현재 설치된 kiwipiepy 버전
  # policy: 프로젝트에서 직접 붙인 토큰화 정책 이름
TOKENIZER_SPEC = {"name": "kiwipiepy", "version": version("kiwipiepy"), "policy": "surface-v1"}
# 검색 토큰에서 제외할 품사 > 마침표·물음표·느낌표, 쉼표·콜론, 인용 부호·괄호 계열, 여는 괄호·닫는 괄호, 줄임표, 붙임표·물결표 계열
PUNCTUATION_TAGS = {"SF", "SP", "SS", "SSO", "SSC", "SE", "SO"}


# 형태소 분석 모델을 프로세스 내에서 한 번만 준비하여 재사용
@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    return Kiwi(num_workers=1)


# 문서·질문에 동일한 토큰화 적용; 숫자·영문·부정 표현과 금융 기호는 보존
def tokenize_text(text: str, *, kiwi: Kiwi | None = None) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("BM25 input must be a nonempty text.")
    analyzer = kiwi if kiwi is not None else _get_kiwi()
    return [token.form.casefold() for token in analyzer.tokenize(text)
            if token.tag not in PUNCTUATION_TAGS and token.form.strip()]


# 같은 순서의 문서들을 토큰화하고 BM25와 저장용 토큰 목록 반환
def build_bm25_index(texts: list[str], *, kiwi: Kiwi | None = None) -> tuple[BM25Okapi, list[list[str]]]:
    tokens = [tokenize_text(text, kiwi=kiwi) for text in texts]
    _validate_tokens(tokens)
    return BM25Okapi(tokens, **BM25_PARAMETERS), tokens


# 역직렬화 시 실행 가능한 pickle 대신 토큰·청크 ID·설정을 JSON으로 저장
def save_bm25_index(
    tokens: list[list[str]], chunk_ids: list[str], file_path: Path, *, fingerprint: str,
) -> None:
    _validate_tokens(tokens)
    if (len(chunk_ids) != len(tokens) or any(not isinstance(key, str) or not key for key in chunk_ids)
            or len(set(chunk_ids)) != len(chunk_ids)):
        raise ValueError("BM25 requires one unique chunk ID per document.")
    payload = {"fingerprint": fingerprint, "chunk_ids": chunk_ids, "tokens": tokens,
               "parameters": BM25_PARAMETERS, "tokenizer": TOKENIZER_SPEC}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


# 저장된 토큰·설정을 검증하고 BM25를 메모리에 재구성하여 메타데이터와 함께 반환
def load_bm25_index(file_path: Path) -> tuple[BM25Okapi, dict]:
    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Invalid BM25 index JSON.")
    tokens, ids, parameters = payload.get("tokens"), payload.get("chunk_ids"), payload.get("parameters")
    _validate_tokens(tokens)
    if (not isinstance(ids, list) or len(ids) != len(tokens)
            or any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids)):
        raise ValueError("BM25 chunk IDs do not match the token corpus.")
    if not isinstance(parameters, dict) or set(parameters) != set(BM25_PARAMETERS):
        raise ValueError("Invalid BM25 parameters.")
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in parameters.values())
            or parameters["k1"] <= 0 or not 0 <= parameters["b"] <= 1 or parameters["epsilon"] < 0):
        raise ValueError("Invalid BM25 parameter values.")
    if payload.get("tokenizer") != TOKENIZER_SPEC:
        raise ValueError("BM25 tokenizer changed; rebuild the news indexes.")
    return BM25Okapi(tokens, **parameters), payload


# 빈 문서·잘못된 토큰 구조로 인한 BM25 계산 오류를 생성·로드 전에 차단
def _validate_tokens(tokens: list[list[str]]) -> None:
    if (not isinstance(tokens, list) or not tokens
            or any(not isinstance(document, list) or not document
                   or any(not isinstance(token, str) or not token for token in document) for document in tokens)):
        raise ValueError("BM25 requires nonempty token lists for all documents.")

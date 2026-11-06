import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.indexing import bm25_index as bm25


# 실제 형태소 모델 대신 토큰 구조만 재현하는 테스트 분석기
class FakeKiwi:
    def tokenize(self, text):
        return [SimpleNamespace(form=part, tag="NNG") for part in text.split()]


# JSON 토큰 저장 후에도 동일 문서 점수·청크 순서로 복원
def test_bm25_roundtrip(tmp_path):
    index, tokens = bm25.build_bm25_index(["반도체 실적", "금리 인상", "환율 상승"], kiwi=FakeKiwi())
    path = tmp_path / "tokens.json"
    bm25.save_bm25_index(tokens, ["a", "b", "c"], path, fingerprint="test")
    restored, metadata = bm25.load_bm25_index(path)
    np.testing.assert_allclose(restored.get_scores(["반도체"]), index.get_scores(["반도체"]))
    assert metadata["chunk_ids"] == ["a", "b", "c"]
    assert metadata["fingerprint"] == "test"


# 실제 Kiwi 모델이 한국어·숫자·부정 표현을 토큰화하는지 확인
def test_real_kiwi_tokenization():
    tokens = bm25.tokenize_text("삼성전자 영업이익은 10조원이고 감소하지 않았다.")
    assert any("삼성" in token for token in tokens)
    assert any("10" in token for token in tokens)
    assert any("않" in token for token in tokens)


# 빈 문서·공백 문서·토큰 없는 문서는 인덱스 계산 전에 실패
@pytest.mark.parametrize("texts", [[], [""], [" "]])
def test_empty_corpus_rejected(texts):
    with pytest.raises(ValueError):
        bm25.build_bm25_index(texts, kiwi=FakeKiwi())


# 서로 다른 문서에 같은 청크 ID를 저장하지 않음
def test_duplicate_chunk_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        bm25.save_bm25_index([["a"], ["b"]], ["same", "same"], tmp_path / "tokens.json", fingerprint="test")


# 손상된 토큰·파라미터는 로드 시 거부
@pytest.mark.parametrize("failure", ["tokens", "parameters", "tokenizer"])
def test_invalid_saved_bm25(tmp_path, failure):
    path = tmp_path / "tokens.json"
    bm25.save_bm25_index([["a"]], ["id"], path, fingerprint="test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[failure] = {} if failure != "parameters" else {"k1": 0, "b": 0.75, "epsilon": 0.25}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        bm25.load_bm25_index(path)

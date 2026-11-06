import numpy as np
import pytest

from src.indexing.faiss_index import build_faiss_index, load_faiss_index, save_faiss_index


# 벡터 정규화·코사인 순위·저장 및 재로딩과 원본 배열 보존 검증
def test_faiss_roundtrip(tmp_path):
    vectors = np.array([[3.0, 4.0], [4.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    original = vectors.copy()
    index = build_faiss_index(vectors)
    np.testing.assert_array_equal(vectors, original)
    np.testing.assert_allclose(index.reconstruct(0), [0.6, 0.8], atol=1e-6)
    path = tmp_path / "news.index"
    save_faiss_index(index, path)
    loaded = load_faiss_index(path)
    scores, rows = loaded.search(np.array([[1.0, 0.0]], dtype=np.float32), 3)
    assert rows[0].tolist() == [1, 0, 2]
    np.testing.assert_allclose(scores[0], [1.0, 0.6, 0.0], atol=1e-6)


# 빈 행렬·잘못된 차원·0벡터·무한대·NaN 거부
@pytest.mark.parametrize("vectors", [[], [1, 2], [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[], []]])
def test_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        build_faiss_index(vectors)


# 큰 유한값을 정규화할 때 float32 제곱 오버플로로 0벡터가 되지 않음
def test_large_vectors_are_normalized():
    index = build_faiss_index([[1e38, 1e38]])
    np.testing.assert_allclose(np.linalg.norm(index.reconstruct(0)), 1.0, atol=1e-6)

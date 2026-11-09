import numpy as np
import pytest
import faiss

from src.indexing.faiss_index import build_faiss_index, load_faiss_index, save_faiss_index
from src.indexing import faiss_index


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


# 생성자만 교체해도 공통 학습·추가·저장·로딩 과정이 IVF·HNSW에서 동작
@pytest.mark.parametrize("kind", ["ivf", "hnsw"])
def test_alternative_index_roundtrip(tmp_path, monkeypatch, kind):
    flat_factory = faiss.IndexFlatIP

    # 주석으로 선택하는 두 대체 생성자와 같은 인덱스를 테스트에 주입
    def alternative_factory(dimension):
        if kind == "ivf":
            return faiss.IndexIVFFlat(flat_factory(dimension), dimension, 1, faiss.METRIC_INNER_PRODUCT)
        return faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)

    monkeypatch.setattr(faiss_index.faiss, "IndexFlatIP", alternative_factory)
    thread_count = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        vectors = np.array([[1, 0], [0.6, 0.8], [0, 1]] * 20, dtype=np.float32)
        index = build_faiss_index(vectors)
        path = tmp_path / "news.index"
        save_faiss_index(index, path)
        loaded = load_faiss_index(path)
        scores, rows = loaded.search(np.array([[1, 0]], dtype=np.float32), 1)
        assert loaded.is_trained and loaded.ntotal == len(vectors)
        assert scores[0][0] == pytest.approx(1)
        assert rows[0][0] % 3 == 0
    finally:
        faiss.omp_set_num_threads(thread_count)

from pathlib import Path

import faiss
import numpy as np


# 원본 벡터는 변경하지 않고 L2 정규화한 뒤 코사인 검색용 내적 인덱스 생성
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    matrix = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError("FAISS requires a nonempty finite matrix without zero vectors.")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("FAISS requires a nonempty finite matrix without zero vectors.")
    matrix /= norms[:, None]
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


# 호출자가 지정한 경로에 FAISS 바이너리 인덱스 저장
def save_faiss_index(index: faiss.Index, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(file_path))


# 저장된 내적 인덱스를 읽고 형식·벡터 개수·차원 검증
def load_faiss_index(file_path: Path) -> faiss.Index:
    index = faiss.read_index(str(file_path))
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or index.d < 1 or index.ntotal < 1:
        raise ValueError("Invalid FAISS cosine index.")
    return index

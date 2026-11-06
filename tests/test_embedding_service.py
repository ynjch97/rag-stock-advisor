from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.embeddings import embedding_service as service


# 묶음 요청 결과를 입력 순서대로 결합하고 float32로 반환
def test_batches_keep_document_order(monkeypatch):
    request = Mock(side_effect=[[[1, 2], [3, 4]], [[5, 6]]])
    monkeypatch.setattr(service, "request_embeddings", request)
    vectors = service.embed_texts(["a", "b", "c"], batch_size=2)
    np.testing.assert_array_equal(vectors, [[1, 2], [3, 4], [5, 6]])
    assert vectors.dtype == np.float32
    assert [call.args[0] for call in request.call_args_list] == [["a", "b"], ["c"]]


# 종목명과 제목을 포함한 동일 텍스트를 두 검색 구조에 제공
def test_news_index_text_contains_context():
    chunk = SimpleNamespace(stock_name="삼성전자", title="실적 발표", content="영업이익 10조원")
    assert service.build_news_index_texts([chunk]) == ["종목: 삼성전자\n제목: 실적 발표\n영업이익 10조원"]


# 부족한 벡터·빈 차원·비정상 수치·차원 불일치는 인덱스 생성 전에 거부
@pytest.mark.parametrize("response", [[], [[1]], [[], []], [[float("nan")], [1]], [[float("inf")], [1]], [[1], [1, 2]]])
def test_invalid_embedding_matrix(monkeypatch, response):
    monkeypatch.setattr(service, "request_embeddings", Mock(return_value=response))
    with pytest.raises(ValueError):
        service.embed_texts(["a", "b"])


# 서로 다른 배치의 차원이 바뀌어도 결합하지 않음
def test_inconsistent_batch_dimensions(monkeypatch):
    monkeypatch.setattr(service, "request_embeddings", Mock(side_effect=[[[1, 2]], [[1, 2, 3]]]))
    with pytest.raises(ValueError, match="dimensions"):
        service.embed_texts(["a", "b"], batch_size=1)


# 잘못된 입력은 API 호출 전에 검증
@pytest.mark.parametrize("texts,batch_size", [([], 32), ([" "], 32), (["a"], 0), (["a"], True), (["a"], 2049)])
def test_invalid_inputs_do_not_request(monkeypatch, texts, batch_size):
    request = Mock()
    monkeypatch.setattr(service, "request_embeddings", request)
    with pytest.raises(ValueError):
        service.embed_texts(texts, batch_size=batch_size)
    request.assert_not_called()

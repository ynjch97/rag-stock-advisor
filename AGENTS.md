# 폴더 구조 및 파일 설명
``` text
README.md                           # 개발 관련 내용
AGENTS.md                           # 개발 규칙
.env                                # 환경변수 관리
.venv                               # 가상환경
main.py                             # 프로그램 시작 파일 (질문을 입력받고 전체 흐름을 실행)
requirements.txt                    # Python 패키지 목록

docs/                               # 개발 관련 문서 정리

data/
  sample/
    sample_stocks.json              # MVP용 샘플 주가 데이터
    sample_news.json                # MVP용 샘플 뉴스 데이터

src/
  __init__.py
```

# 질문 유형
- A 종목 사도 될까?
- A 종목 하락하는 이유 뭐야?

# 테스트
- `pytest` 사용

<!--
최종 폴더 구조 예상
(추후 변동될 가능성 있음)

```
main.py                              # CLI 기반 MVP 실행 진입점

.env                                 # 환경변수 관리
requirements.txt                     # Python 패키지 목록
README.md                            # 개발 관련 내용
AGENTS.md                            # 프로젝트 규칙

data/
  sample/
    sample_stocks.json               # MVP용 샘플 주가 데이터
    sample_news.json                 # MVP용 샘플 뉴스 데이터

  raw/
    prices/                          # 수집한 원본 주가 데이터
    news/                            # 수집한 원본 뉴스 데이터
    reports/                         # 수집한 원본 리포트 데이터
    disclosures/                     # 수집한 원본 공시 데이터
    financials/                      # 수집한 원본 재무 데이터

  processed/
    prices/                          # 전처리된 주가 데이터
    news/                            # 전처리된 뉴스 데이터
    reports/                         # 전처리된 리포트 데이터
    disclosures/                     # 전처리된 공시 데이터
    financials/                      # 전처리된 재무 데이터

  chunks/
    news/                            # 청킹된 뉴스 문서
    reports/                         # 청킹된 리포트 문서
    disclosures/                     # 청킹된 공시 문서
    financials/                      # 청킹된 재무 문서

  vector_store/
    faiss/
      stock_advisor.index            # FAISS 벡터 인덱스 파일
      stock_advisor_metadata.json    # FAISS 벡터와 원문 청크 매핑 메타데이터

src/
  __init__.py

  config/
    __init__.py
    settings.py                      # .env 로딩 및 앱 설정 관리
    paths.py                         # data/raw, processed, chunks, vector_store 경로 관리

  app/
    __init__.py
    stock_advice_workflow.py         # 질문 분석부터 최종 응답까지 전체 흐름 실행
    rag_indexing_workflow.py         # 수집/전처리/청킹/임베딩/FAISS 저장 흐름 실행

  domain/
    __init__.py
    stock_models.py                  # 주가 데이터 구조
    news_models.py                   # 뉴스 데이터 구조
    document_models.py               # 원문/전처리/청크 문서 구조
    embedding_models.py              # 임베딩 결과 데이터 구조
    search_models.py                 # 검색 결과, Hybrid Search 결과 구조
    advice_models.py                 # 질문 분석 결과, 최종 응답 구조

  data_collection/
    __init__.py
    stock_price_collector.py         # 주가/시세 데이터 수집
    news_collector.py                # 뉴스 데이터 수집
    report_collector.py              # 증권사 리포트/전망 데이터 수집
    disclosure_collector.py          # 공시 데이터 수집

  data_loaders/
    __init__.py
    sample_stock_loader.py           # data/sample/sample_stocks.json 읽기
    sample_news_loader.py            # data/sample/sample_news.json 읽기
    sample_report_loader.py          # data/sample/sample_reports.json 읽기
    json_document_loader.py          # JSON 문서 로딩 공통 기능

  preprocessing/
    __init__.py
    text_cleaner.py                  # 본문 정제, 불필요 문자 제거
    document_normalizer.py           # 문서 필드 표준화
    metadata_extractor.py            # 종목명, 날짜, 출처 등 메타데이터 추출

  chunking/
    __init__.py
    document_chunker.py              # 문서 청킹
    chunk_strategy.py                # 토큰/문단/문장 기준 청킹 전략

  embeddings/
    __init__.py
    embedding_service.py             # 텍스트 임베딩 생성
    embedding_model_client.py        # OpenAI/로컬 임베딩 모델 클라이언트

  vector_db/
    __init__.py
    vector_store_client.py           # Vector DB 공통 인터페이스
    faiss_vector_store.py            # FAISS 인덱스 생성, 저장, 로드, 검색
    vector_indexer.py                # 청크 문서 임베딩 후 FAISS에 저장

  search/
    __init__.py
    keyword_search.py                # 키워드/BM25 검색
    vector_search.py                 # FAISS 벡터 유사도 검색
    hybrid_search.py                 # Keyword Search + FAISS Vector Search 결합
    search_result_ranker.py          # 검색 결과 재정렬

  parsers/
    __init__.py
    stock_question_parser.py         # 사용자 질문에서 종목명, 의도, 기간 추출

  services/
    __init__.py
    stock_context_service.py         # 주가/뉴스/Hybrid Search 결과를 분석 컨텍스트로 구성
    stock_advice_service.py          # 자연어 투자 분석 응답 생성

  agents/
    __init__.py
    stock_advisor_agent.py           # 최종 투자 판단 지원 Agent
    data_retrieval_agent.py          # FAISS/키워드 검색 및 데이터 조회 담당 Agent
    risk_analysis_agent.py           # 리스크 요인 분석 Agent
```
-->
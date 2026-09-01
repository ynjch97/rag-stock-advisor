# RAG Stock Advisor
- 뉴스·전망·시세 데이터를 종합 분석하는 RAG 기반 주식 투자 의사결정 지원 시스템
- 실행 : `python -X utf8 main.py --mode cli`

## 1. 설계

### 1-1. 동작 흐름
``` text
사용자 질문 ("삼성전자 지금 매수해도 돼?")
↓
질문 분석 (삼성전자 / 매수 판단 / 현재 시점)
↓
데이터 수집 (실시간 주가, 차트, 뉴스, 공시, 재무, 수급)
↓
분석 (주가 흐름 + 뉴스 요약 + 전문가 의견)
↓
LLM 응답 ("지금 매수는 신중, 이유는…")
```

### 1-2. 기술 스택
- 언어 : Python
- 개발환경 : VS Code

### 1-3. 개발 환경 구축
- 가상환경(.venv) 만들기 : `python -m venv .venv`
- 가상환경 활성화 : `.venv\Scripts\activate`
- 필요 라이브러리 설치
``` bash
pip install openai
pip install fastapi
```
- `requirements.txt`에 저장 : `pip freeze > requirements.txt`
- `requirements.txt` 내용대로 설치 : `pip install -r requirements.txt`

### 1-4. MVP 설계
``` text
사용자 질문 (stock_advice_workflow.py)
↓
질문에서 종목명/의도 추출 (stock_question_parser.py)
↓
주가/뉴스 정보 가져오기 (sample_stock_loader.py, sample_news_loader.py)
↓
주가/뉴스 컨텍스트 구성 (stock_context_service.py)
↓
AI가 자연어로 간단 분석 응답 (stock_advice_service.py)
```

## 2. 데이터 설계

### 2-1. 데이터 흐름
- 원본 데이터 `data/raw/` -> 전처리 `data/processed/` -> 청킹 `data/chunks/` -> 임베딩 및 FAISS 벡터 인덱스 `data/vector_store/faiss/`

### 2-2. 주가 데이터
- JSON 필드 구조 (raw 데이터 기준)
``` json
[
  # 거래일별 가격/거래량 원본 데이터
  # data/raw/prices/005930_daily.json
  {
    "stock_name": "삼성전자",
    "stock_code": "005930",
    "date": "2026-07-16",     # 거래일
    "open": 79300,            # 시가
    "high": 79600,            # 고가
    "low": 78000,             # 저가
    "close": 78200,           # 종가
    "volume": 16800000        # 거래량
  }
]
```
- JSON 필드 구조 (processed 데이터 기준)
``` json
[
  # data/processed/prices/005930_trend.json
  {
    "stock_name": "삼성전자",
    "stock_code": "005930",
    "base_date": "2026-07-07",
    "recent_days": 5,             # 최근 몇 거래일 기준인지
    "recent_return_rate": -1.875, # recent_days 동안의 수익률
    "up_days": 1,                 # 상승 거래일 수
    "down_days": 4,               # 하락 거래일 수
    "trend": "최근 5거래일 약세"    # 사람이 읽기 쉬운 가격 흐름 요약
  }
]
```
- JSON 필드 구조 (sample 데이터 기준)
``` json
[
  {
    "stock_name": "삼성전자",        # 종목명
    "stock_code": "005930",         # 종목코드
    "current_price": 78500,         # 현재가
    "change_rate": -1.2,            # 등락률
    "trend": "최근 5거래일 약세",     # 최근 흐름 요약 (원본 데이터를 계산하여 생성)
    "base_date": "2026-07-07"      # 시세 기준일
  }
]
```

#### 2-2-1. 주가 실시간 데이터
- 실시간 현재가를 API 직접 조회
- Vector DB와 raw 데이터 모두 저장하지 않음
- JSON 필드 구조
``` json
{
  "stock_name": "삼성전자",
  "stock_code": "005930",
  "current_price": 78500,
  "change_rate": -1.2,
  "base_date": "2026-07-07"
}
```

### 2-3. 뉴스 데이터
- JSON 필드 구조 (sample 데이터 기준)
``` json
[
  {
    "stock_name": "삼성전자",                                    # 종목명
    "stock_code": "005930",                                     # 종목코드
    "title": "삼성전자 반도체 업황 회복 기대",                      # 뉴스 제목
    "summary": "메모리 반도체 가격 회복 가능성이 제기되고 있습니다.",  # 뉴스 요약
    "sentiment": "positive",                                    # 뉴스 감성: positive / neutral / negative
    "published_date": "2026-07-07"                              # 뉴스 발행일
  }
]
```

## 3. 데이터 수집 및 전처리

### 3-1. RAG 활용
- Vector DB에 저장 후 RAG 검색
- 대상 : 공시 문서, 기업 사업보고서, 뉴스 기사 요약, 리포트 요약, 종목별 과거 이슈
- 현재가는 API로 직접 실시간 데이터 조회

## 4. 시스템 아키텍처

### 4-1. User Query
- 사용자의 자연어 질문이 입력되는 단계
- "삼성전자 왜 올랐어?" -> 이후 모듈에서 분석 가능하도록 의미 단위로 해석됨

### 4-2. Query Analyzer
- 종목명 등의 핵심 엔티티 분석 및 질의 의도 추출
- 비정형 자연어를 검색에 적합한 형태로 변환 + 필요 시 키워드 확장(Query Rewriting)

### 4-3. Agent Controller
- 질의의 복잡도를 판단하고, 어떤 데이터를 어떤 순서로 검색할지 전략을 결정하는 핵심 모듈
  - *주가 하락 원인 : 최근 주가 흐름 확인 → 관련 뉴스/공시 검색 → 원인 후보 정리*
  - *매수 판단 질문 : 현재가/추세 확인 → 뉴스/리포트 검색 → 리스크/긍정 요인 비교*
- 중간 결과를 바탕으로 추가 검색을 수행하는 멀티 스텝 Retrieval 및 Reasoning -> 단순 RAG를 넘어서는 Agentic RAG 구조를 구현

### 4-4. Retriever Layer
- 뉴스 데이터
  - Embedding Model을 통해 벡터화되어 Vector DB에 저장됨
  - 유사도 계산(Cosine Similarity 등)을 통해 관련 문서 검색
  - 하이브리드 검색(Hybrid Search) 적용
  - BM25 알고리즘을 통한 정확한 키워드 일치 검색 + 코사인 유사도(Cosine Similarity)를 활용한 Vector Search
    - BM25 : "삼성전자", "주가", "인수"
    - Vector Search : "주가 왜 올랐어?" <-> "매수세 증가", "계약 체결" 연결
  - 이후 `공시 문서, 기업 사업보고서` 등의 **텍스트 문서 데이터** 추가 예정
- 주가 데이터
  - 정형 데이터이므로 벡터 검색이 아닌 거래일 조건 기반 필터링 및 시계열 조회 방식

### 4-5. Context Builder
- LLM 입력용 컨텍스트 구성
- 주가/뉴스 데이터를 LLM 입력에 적합한 순서로 정렬하고 재구성
  - *향후 고도화 방안 : 지식 그래프 구조를 활용하여 데이터 간 관계를 정렬하고, 기준에 따라 순서를 재배열(Re-ranking)*
- 불필요한 정보는 제거하고 핵심 정보만 유지 -> LLM이 정확한 추론을 수행하도록 함

### 4-6. LLM Generator
- 구성된 컨텍스트를 기반으로 뉴스와 주가 간의 관계를 분석하고 자연어 응답을 생성
- 단순 요약이 아닌 원인(뉴스/공시/리포트 등) → 결과(주가) 구조
- 원본 검색 데이터와 대조하는 검증 절차를 포함해 Hallucination을 최소화하고 신뢰성을 확보

<!--
```
Hybrid Search 흐름:
src/parsers/stock_question_parser.py
↓
src/search/keyword_search.py
↓
src/search/vector_search.py
↓
src/search/hybrid_search.py
↓
src/search/search_result_ranker.py
↓
src/agents/data_retrieval_agent.py
↓
src/agents/stock_advisor_agent.py
```
-->
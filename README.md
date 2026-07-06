# RAG Stock Advisor
- 뉴스·전망·시세 데이터를 종합 분석하는 RAG 기반 주식 투자 의사결정 지원 시스템

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

## 2. MVP 설계
``` text
사용자 질문 (stock_advice_workflow.py)
↓
질문에서 종목명/의도 추출 (stock_question_parser.py)
↓
주가/뉴스 정보 가져오기 (sample_stock_loader.py, sample_news_loader.py)
↓
AI가 자연어로 간단 분석 응답 (stock_context_service.py, stock_advice_service.py)
```

## 3. 데이터 설계

### 3-1. 주식 데이터
- JSON 필드 구조 (sample 데이터 기준)
``` json
[
  {
    "stock_name": "삼성전자",        # 종목명
    "stock_code": "005930",         # 종목코드
    "current_price": 78500,         # 현재가
    "change_rate": -1.2,            # 등락률
    "trend": "최근 5거래일 약세",     # 최근 흐름 요약
    "base_date": "2026-07-07"      # 시세 기준일
  }
]
```

### 3-2. 뉴스 데이터
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

### 3-3. 데이터 흐름
- 원본 데이터 `data/raw/` -> 전처리 `data/processed/` -> 청킹 `data/chunks/` -> 임베딩 및 FAISS 벡터 인덱스 `data/vector_store/faiss/`

## 4. 데이터 수집 및 전처리

#### 4-1. RAG 활용
- Vector DB에 저장 후 RAG 검색
- 대상 : 공시 문서, 기업 사업보고서, 뉴스 기사 요약, 리포트 요약, 종목별 과거 이슈
- 현재가 등의 실시간 데이터는 API로 직접 조회

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
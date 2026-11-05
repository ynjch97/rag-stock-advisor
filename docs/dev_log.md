### 1. 프로젝트 샘플 모듈 생성

#### Data Loader
- `paths.py`
- `sample_stock_loader.py`
  - data/sample/sample_stocks.json 읽기
  - 종목명 기준으로 가장 최근 base_date 데이터 반환
- `sample_news_loader.py`
  - data/sample/sample_news.json 읽기
  - 종목명 기준 뉴스 목록 반환
``` bash
# powershell 테스트
python -c "from src.data_loaders.sample_stock_loader import load_latest_sample_stock; print(load_latest_sample_stock('삼성전자'))"
python -c "from src.data_loaders.sample_news_loader import load_recent_sample_news; print(load_recent_sample_news('삼성전자', limit=3))"
```

#### Question Parser
- `stock_question_parser.py`
  - 사용자 질문에서 stock_name, intent 추출
  - EX) "삼성전자 사도 될까?" (stock_name: 삼성전자 / intent: buy_opinion)
- 샘플 단계 개발
  - 질문에서 종목명 추출 / 질문 의도 분류 / workflow로 넘기기
- 추후 개선 방향
  - 종목명/종목코드 매핑 테이블 분리, 별칭 처리('삼성전자' = '삼전')
  - 형태소 분석 또는 LLM 기반 intent 분류
  - 기간 추출: 최근 1주일, 한 달, 올해
``` bash
# powershell 테스트
python -c "from src.parsers.stock_question_parser import parse_stock_question; print(parse_stock_question('삼성전자 지금 사도 될까?'))"
```

#### Context Construction
- `stock_context_service.py`
  - 주가 데이터 + 뉴스 데이터를 이용해 하나의 분석 컨텍스트 생성

#### Generated Response
- `stock_advice_service.py`
  - 샘플 단계에서는 OpenAI 호출 없이 규칙 기반 자연어 응답 생성

#### 조립 및 연결
- `stock_advice_workflow.py`
  - 위 모듈들을 순서대로 호출해서 MVP 전체 흐름 조립
- `main.py`
  - 질문 입력 → workflow 실행 → 응답 출력

#### pytest
- pytest로 최소 테스트 추가

### 2. 전체 종목 수집

#### stock_name_code.json
- 주가 데이터 매핑 정보 저장 (`?market=KOSPI`, `?market=KOSDAQ` 조회 데이터 모두 한 파일에 저장 및 조회)
- `stock_reference_collector.py` : 전체 종목 수집 및 매핑 파일 저장·갱신
- `stock_reference_loader.py` : 저장된 매핑 파일 로딩
- `paths.py` : 매핑 파일 경로 추가

#### stock_question_parser.py
- `sample_stock_loader` 대신 `stock_reference_loader` 로 변경

### 3. 주가 데이터 수집/전처리

#### 모델 정의
- `stock_models.py` : 실시간 시세, 일별 OHLCV, 주가 추세 각각의 클래스 정의
  - RealtimeStock : 실시간 시세
  - DailyStock : 일별 OHLCV 저장
  - StockTrend : 주가 추세 정보

#### Data Collector
- `stock_token_manager.py`
  - 주가 데이터 수집용 토큰 발급
- `stock_collector.py`
  - `/api/v1/market-calendar/KR`로 데이터 조회 기준 파악
    - 실시간 데이터 조회 여부 및 일별 OHLCV 조회 기준일
  - `stock_reference_loader.py` -> 질문과 일치하는 종목코드/종목명 매핑
    - 장중에는 RealtimeStock 실시간 주가 조회 (저장 X)
    - DailyStock 데이터 존재하지 않으면 조회 -> `data/raw/prices/[종목코드]_daily.json`에 저장
      - 등락률 계산법 : `(당일종가 - 전일종가) / 전일종가 * 100` *(증권사는 전일종가 대신 기준가를 사용하지만 토스증권에서는 가져올 수 없으므로, 해당 정보를 할용하여 분석하되 사용자 응답에는 노출하지 않음)*
    - StockTrend 데이터 존재하지 않으면 누락된 거래일 추가 조회 -> `data/processed/prices/[종목코드]_trend.json`에 저장
- `[종목코드]_daily.json`
  - 같은 날짜는 교체하고, 날짜 순으로 정렬하여 저장

#### Data Preprocessing
- `stock_preprocessor.py`
  - 주가 데이터 전처리 후 StockTrend 저장
- `[종목코드]_trend.json`
  - 일봉 5개로 수익률, 상승/하락 횟수, 추세 요약 등을 계산/분석하여 저장
  - 같은 날짜는 교체하고, 날짜 순으로 정렬하여 저장

#### 조립 및 연결
- `stock_advice_workflow.py`
  - 조회·전처리 호출 추가
  - `STOCK_API_PROVIDER=toss`로 설정하여 실제 조회 사용하도록 함
  - RealtimeStock, DailyStock, StockTrend 를 기존 컨텍스트와 응답 생성 모듈에 전달

#### 테스트
- 휴장일, 장 시작 직전, 장중, 장 종료 시각 이후 등의 시점별 테스트 필요

### 4. 질문 분석

#### Question Parser
- `stock_question_parser.py`, `llm_question_analyzer.py`
  - LLM 기반으로 비정형 표현의 의도/기간 추출
  - `ParsedQuestion` 확장
    - `original_question` : 원본 질문
    - `stocks` : 단/복수 종목 목록 (원문 표현, 정식 종목명 또는 코드)
    - `intent` : 매수 판단, 상승/하락 원인, 종목 비교 등
    - `period` : 거래일 수, 기간, 명시적 날짜 범위 구분
    - `needs_clarification` : 종목이나 기간이 모호한지 여부
    - `clarification_question` : 필요한 확인 질문
``` json
{
  "original_question": "삼전이 하이닉스보다 오르는 이유는?",
  "stocks": [
    {
      "mention": "삼전",
      "stock_name": "삼성전자",
      "stock_code": "005930",
      "role": "target"
    },
    {
      "mention": "하이닉스",
      "stock_name": "SK하이닉스",
      "stock_code": "000660",
      "role": "comparison"
    }
  ],
  "intent": "rise_reason",
  "period": {
    "kind": "default",
    "recent_days": 5,
    "start_date": null,
    "end_date": null
  },
  "needs_clarification": false,
  "clarification_question": null
}
```
- 응답 지연 개선
  - `OPENAI_REASONING` 값을 기본값에서 `minimal` 로 변경
  - OpenAI 호출 시도 횟수 확인을 위해 `openai_call_attempts` 값 추가
    - 추후 문제 발생 시 확인을 위함 (현재는 1회 실행)

### 5. 뉴스 데이터 수집/전처리

#### 모델 정의
- `news_models.py`

#### Data Collector
- `news_collector.py`
  - `NEWS_PUBLISHERS` 에 해당하는 언론사이면 뉴스 원문 추출
    - 전처리 및 청킹 시 `original_article.body` 값 사용 + 우선으로 분석에 활용함
    - `description` 값으로 대체하는 경우 : 이외의 언론사(`unsupported_publisher`), 설정한 기간을 벗어난(`outside_period`) 데이터 등

#### Data Preprocessing
- `news_preprocessor.py`
  - `NEWS_PUBLISHERS` 에 해당하는 언론사이면 중요도 `O`

#### Data Chunking
- `document_chunker.py`
  - 추후 공시 문서, 기업 사업보고서 등에도 사용 예정

#### Data Embedding
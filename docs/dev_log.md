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
  - `sample_stock_loader.py` 대신 실제 데이터 반영
  - 질문에 대한 주가 데이터 수집 -> RealtimeStock 실시간 주가 조회 + DailyStock 데이터 저장
- 데이터 조회
  - `/api/v1/candles?symbol=005930&interval=1d&count=1&before=2026-10-07T23:59:59+09:00`
    - 종목 번호, 조회일 데이터 변수로 활용
- 데이터 저장
  - 정규 시장 전/후 : 당일 마감/직전 영업일 데이터 조회 후 DailyStock 저장
  - 정규 시장 중 : 실시간 데이터 조회 (저장 X)
  - 휴장일 판단은 `/api/v1/market-calendar/KR` API 조회 기준으로 함

#### Data Preprocessing
- `stock_preprocessor.py`
  - 주가 데이터 전처리 후 StockTrend 저장
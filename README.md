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
- 언어 : `Python`
- 개발환경 : `VS Code`
- 사용자 질의 분석 LLM : `OpenAI gpt-5-mini`
- 데이터 청킹 : `LangChain RecursiveCharacterTextSplitter`
- 텍스트 임베딩 : `OpenAI text-embedding-3-small`
- 벡터 인덱싱, 유사도 검색 : `FAISS`
- 한국어 형태소 분석·토큰화 : `Kiwi`
- 키워드 검색 : `BM25`

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
- 실시간 데이터 : API로 실시간 조회하고 별도 저장하지 않음
- 일별 OHLCV : API에서 수집하여 `data/raw/prices/`에 저장
  - 장 마감 후 하루에 한 번 정도 갱신하는 것으로 함
- 추세 정보 : OHLCV를 계산하여 `data/processed/prices/`에 저장
  - 데이터의 최신 여부를 판단하여, 없거나 오래되었으면 새롭게 전처리 실행

#### 2-2-1. 주가 데이터
- JSON 필드 구조 (raw 데이터 기준)
``` json
[
  # 거래일별 가격/거래량 원본 데이터
  # data/raw/prices/005930_daily.json
  {
    "stock_name": "삼성전자",
    "stock_code": "005930",
    "base_date": "2026-07-16", # 거래일
    "open": 79300,            # 시가
    "high": 79600,            # 고가
    "low": 78000,             # 저가
    "close": 78200,           # 종가
    "volume": 16800000,       # 거래량
    "change_rate": 4.1,       # 등락률(%) -> (당일종가 - 전일종가) / 전일종가 * 100
    "change_price": 9000      # 등락금액
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
    "recent_change_rate": -1.875, # recent_days 동안의 수익률
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

#### 2-2-2. 주가 실시간 데이터
- 실시간 현재가를 API 직접 조회
- Vector DB와 raw 데이터 모두 저장하지 않음
- JSON 필드 구조
``` json
{
  "stock_name": "삼성전자",
  "stock_code": "005930",
  "current_price": 78500,
  "change_rate": -1.2,            # 등락률(%) -> (현재가 - 전일종가) / 전일종가 * 100
  "base_date": "2026-07-07"
}
```

### 2-3. 뉴스 데이터
- JSON 필드 구조 (raw 데이터 기준)
``` json
{
  # 네이버 API 응답 결과 json + 원문 조회 결과 필드 추가
  "title": "네이버 뉴스 API 제목",
  "originallink": "https://www.mt.co.kr/...",
  "link": "https://n.news.naver.com/mnews/article/...",
  "description": "네이버 뉴스 API 요약 결과",
  "pubDate": "Thu, 05 Nov 2026 20:30:00 +0900",
  "original_article": {
    "url": "https://www.mt.co.kr/...",
    "publisher": "머니투데이",
    "status": "success",
    "html": "뉴스 HTML 전체",
    "body": "뉴스 본문 전체",
    "error": null,
    "final_url": "https://www.mt.co.kr/..."
  }
},
```
- JSON 필드 구조 (processed 데이터 기준)
``` json
[
  {
    "stock_name": "삼성전자",                                    # 종목명
    "stock_code": "005930",                                     # 종목코드
    "url": "https://www.fnnews.com/news/202611030838053580",    # 뉴스 URL
    "title": "삼성전자 반도체 업황 회복 기대",                      # 뉴스 제목
    "content": "메모리 반도체 가격 회복 가능성이 제기되고 있습니다.",  # 뉴스 원문or요약
    "important": "O",                                           # 뉴스 중요도 (원문 유무에 따라 O/X)
    "publisher": "mt.co.kr",                                    # 뉴스 발행사
    "published_date": "2026-07-07"                              # 뉴스 발행일
  }
]
```

## 3. 데이터 수집 및 전처리

### 3-1. RAG 활용
- Vector DB에 저장 후 RAG 검색
- 대상 : 공시 문서, 기업 사업보고서, 뉴스 기사 요약, 리포트 요약, 종목별 과거 이슈
- 현재가는 API로 직접 실시간 데이터 조회

### 3-2. 주가 데이터
- 토스증권 API
  - https://developers.tossinvest.com/docs
  - 키움증권, 한국투자증권 API와 비교했을 때, API 구조와 문서가 가장 깔끔하고 국내·미국 주식의 현재가와 일봉을 모두 조회할 수 있어 선정
  하였음
  - OAuth 2.0 액세스 토큰 사용
    - 한 Client에서 유효한 토큰은 1개 (24시간)
    - 새 토큰을 발급하면 기존 토큰은 즉시 무효화

#### 3-2-1. 토스증권 API 사용
- 클라이언트 등록
  - 토스증권 WTS(https://www.tossinvest.com/)에 로그인 후 설정
  - Open API 메뉴에 진입하여 `client_id` 와 `client_secret` 을 발급
- 허용 IP 등록
  - 설정 > Open API 메뉴 하단의 허용 IP 관리 에서 API 호출을 허용할 IP 를 등록
  - 등록된 허용 IP 목록에 없는 IP 에서의 호출은 403 으로 차단됨
- 액세스 토큰 발급
  - `POST /oauth2/token` 으로 Client Credentials Grant 방식의 access token 을 발급
``` bash
$body = @{
    grant_type    = "client_credentials"
    client_id     = "xxx"
    client_secret = "xxx"
}

$response = Invoke-RestMethod `
    -Method Post `
    -Uri "https://openapi.tossinvest.com/oauth2/token" `
    -ContentType "application/x-www-form-urlencoded" `
    -Body $body

$response.access_token
```
- API 호출
  - 발급받은 토큰을 `Authorization: Bearer {access_token}` 헤더에 담아 호출
  - 계좌·자산, 주문, 조건주문 카테고리는 `X-Tossinvest-Account: {accountSeq}` 헤더도 함께 전달
``` bash
# curl 문법
curl.exe -s "https://openapi.tossinvest.com/api/v1/prices?symbols=005930" `
  -H "Authorization: Bearer $($response.access_token)"

# PowerShell 문법
$result = Invoke-RestMethod `
    -Method Get `
    -Uri "https://openapi.tossinvest.com/api/v1/prices?symbols=005930" `
    -Headers @{ Authorization = "Bearer $($response.access_token)" }

$result
```

#### 3-2-2. API 호출 목록 및 결과
- 호가 조회 (`GET /api/v1/orderbook`)
  - ``
- 현재가 조회 (`GET /api/v1/prices`)
  - 여러 종목 조회 : `?symbols=005930%2C000660`
  - `{"result":[{"symbol":"005930","timestamp":"2026-10-02T19:59:59.000+09:00","lastPrice":"276000","currency":"KRW"}]}`
- 상/하한가 조회 (`GET /api/v1/price-limits`)
  - ``
- 캔들(OHLCV) 차트 조회​ (`GET /api/v1/candles`)
  - `?symbol=005930&interval=1d&count=1&before=2026-10-07T23:59:59+09:00`
  - 일봉, count로 한 건만 조회, 지정 날짜의 데이터 조회
  - `{"result":{"candles":[{"timestamp":"2026-10-07T00:00:00.000+09:00","openPrice":"272000","highPrice":"279500","lowPrice":"268000","closePrice":"269000","volume":"24741933","currency":"KRW"}],"nextBefore":"2026-10-06T00:00:00.000+09:00"}}`
``` json
{
  "result": {
    "candles": [
      {
        "timestamp": "2026-03-25T00:00:00+09:00",
        "openPrice": "71600",
        "highPrice": "72300",
        "lowPrice": "71500",
        "closePrice": "72000",
        "volume": "3521000",
        "currency": "KRW"
      },
      {
        "timestamp": "2026-03-24T00:00:00+09:00",
        "openPrice": "71200",
        "highPrice": "71800",
        "lowPrice": "71000",
        "closePrice": "71600",
        "volume": "2984000",
        "currency": "KRW"
      }
    ],
    "nextBefore": "2026-03-24T00:00:00+09:00"
  }
}
```
- 매수 유의사항 조회 (`GET /api/v1/stocks/{symbol}/warnings`)
  - 정리매매, 과열, 투자경고/위험, VI, 신주인수권 등
  - ``
- 투자자별 매매동향 조회 (`GET /api/v1/stocks/{symbol}/investor-trading`)
  - 개인·외국인·기관·기타법인 일별 거래량, 기관 세부 분류 포함
``` json
{
  "result": {
    "nextUntil": "2026-09-16",
    "records": [
      {
        "date": "2026-10-02", "updatedAt": "2026-10-05T04:00:22.000+09:00",
        "individual": {"buyVolume": "968374", "sellVolume": "1461113", "netBuyVolume": "-492739"},
        "foreigner": {"buyVolume": "668699", "sellVolume": "770512", "netBuyVolume": "-101813"},
        "institution": {
          "buyVolume": "761169", "sellVolume": "657739", "netBuyVolume": "103430",
          "breakdown": {
            "financialInvestment": {"buyVolume": "248551", "sellVolume": "211301", "netBuyVolume": "37250"},
            "insurance": {"buyVolume": "7031", "sellVolume": "1326", "netBuyVolume": "5705"},
            "trust": {"buyVolume": "32926", "sellVolume": "36433", "netBuyVolume": "-3507"},
            "privateEquityFund": {"buyVolume": "35033", "sellVolume": "13844", "netBuyVolume": "21189"},
            "bank": {"buyVolume": "1151", "sellVolume": "504", "netBuyVolume": "647"},
            "otherFinancialInstitution": {"buyVolume": "205", "sellVolume": "43", "netBuyVolume": "162"},
            "pensionFund": {"buyVolume": "436272", "sellVolume": "394288", "netBuyVolume": "41984"}
          }
        },
        "otherCorporation": {"buyVolume": "507034", "sellVolume": "16236", "netBuyVolume": "490798"},
        "foreignerHolding": {"holdingQuantity": "363517899", "limitQuantity": "730492365", "holdingRate": "0.4976"},
        "cfd": {"buyBalanceQuantity": "234707", "buyBalanceRate": "0.0003", "sellBalanceQuantity": "3", "sellBalanceRate": "0"}
      }
    ]
  }
}
```

<!-- 
테스트 완료 후 지울 주석 부분임
curl.exe -s "https://openapi.tossinvest.com/api/v1/candles?symbols=005930&interval=1m" `
  -H "Authorization: Bearer $($response.access_token)"
curl.exe -s "https://openapi.tossinvest.com/api/v1/prices?symbols=005930%2C000660" `
  -H "Authorization: Bearer $($response.access_token)"
curl.exe -s "https://openapi.tossinvest.com/api/v1/market-calendar/KR" `
  -H "Authorization: Bearer $($response.access_token)"  
curl.exe -s "https://openapi.tossinvest.com/api/v1/stocks/000660/investor-trading" `
  -H "Authorization: Bearer $($response.access_token)"

stock_collector.py 구현
stock_question_parser.py 
종목명 추출, 종목코드로 변환 후 조회
-->

#### 3-2-3. 전체 종목 데이터 저장
- 사용자 쿼리에서 추출한 종목명과 종목코드를 매핑하기 위함
  - 미리 수집된 데이터에서 매핑되는 코드 확보
  - 파일이 없거나, 종목명이 존재하지 않으면 데이터 갱신 -> `/data/reference/stock_name_code.json`
- 마켓별 전체 종목 조회 (초당 최대 1회 사용 가능)
``` bash
curl.exe -s "https://openapi.tossinvest.com/api/v1/stocks/all?market=KOSPI" `
  -H "Authorization: Bearer $($response.access_token)"
curl.exe -s "https://openapi.tossinvest.com/api/v1/stocks/all?market=KOSDAQ" `
  -H "Authorization: Bearer $($response.access_token)"
```

#### 3-2-4. 주가 데이터 조회 기준
- `/api/v1/market-calendar/KR` API 조회 기준으로 함
  - 정규장 종료의 기준은 `today.integrated.regularMarket.endTime`을 기준으로 함
- 캔들(OHLCV) 차트 조회 활용​ (`GET /api/v1/candles`)
  - 종목 번호, 조회일 데이터를 변수로 활용
  - 장중에는 현재가 조회도 가능 (`GET /api/v1/prices`)
- `today.integrated == null` 이면 휴장일이므로, `previousBusinessDay.date`의 마감 데이터
- `today.integrated != null`
  - 현재 시각 < 정규장 시작 :	`previousBusinessDay.date`의 마감 데이터
  - 정규장 시작 ≤ 현재 시각 < 정규장 종료 : 실시간 데이터 (저장 X)
  - 현재 시각 ≥ 정규장 종료 : `today.date`의 마감 데이터
- 실시간 데이터를 제외한 주가 데이터는 **DailyStock(일별 OHLCV)**으로 저장
  - 필요 시 전처리 후 **StockTrend(추세 정보)** 저장도 함께 실행

#### 3-2-5. 주가 데이터 전처리 기준
- 해당 종목의 StockTrend 파일이 없거나, 새로운 거래일의 DailyStock이 추가되었을 경우 전처리 실행
- 전처리 데이터는 최근 n거래일을 기준으로 추세를 분석하여 생성 (일봉 n개를 의미)
  - 장중 실시간 데이터는 저장하지 않음
  - 직전 완료 거래일까지의 DailyStock이 해당됨 (정규장 종료 후에는 당일)
  - DailyStock에 최근 5거래일의 데이터 중 누락된 데이터가 있다면 모두 조회 필요
  - `recent_change_rate`(수익률) : `rate = (마지막 종가 / 첫 번째 종가 - 1) × 100`
  - `trend` : 최근 n거래일 하락/상승
  - `up_days`, `down_days` : 하락/상승일 카운트

### 3-3. 뉴스 데이터
- 네이버 뉴스 API
  - https://developers.naver.com/docs/serviceapi/search/news/news.md
  - 수집 방식 : 뉴스 API JSON 타입으로 수집
  - 파라미터 : `sort=date, display=30, start=1`
- 사용자 질의의 종목·기간 조건으로 검색 범위를 제한
- 경제/금융 뉴스 위주의 수집 (한국경제, 매일경제, 서울경제, 이데일리, 연합뉴스, 머니투데이 등)
- 기업 실적/공시, 증시 동향, 금리/환율, 정부 정책, 산업별 이슈, 글로벌 경제 동향, 애널리스트 전망 등 주가에 영향을 미치는 뉴스로 구성
- 수집 데이터 : 기사 URL, 제목, 본문, 발행일

#### 3-3-1. 뉴스 데이터 전처리 전략
- 결측치 처리 : 제목, 본문 누락 등 불완전한 기사 제거
- 텍스트 정제
  - 불필요한 HTML 태그 제거, 광고성 문구 및 특수문자 제거
  - 기업명, 금융 수치, 부정 표현 등 의미 분석에 필요한 정보는 보존
- 메타데이터 표준화: 언론사, 날짜, 종목 코드 표준화
- 뉴스 URL 기준으로 중복 기사 제거 + 유사 기사 필터링(제목/본문 유사도 및 발행일 기준)
- `important` : 뉴스 중요도 (설정 기간 내 경제/금융 뉴스를 우선함)

#### 3-3-2. 뉴스 데이터 청킹
- 데이터 수집/갱신 시 수행
- Recursive Chunking (문단·문장 경계를 우선) 
  - 문맥 보존(주가 상승 원인과 기업명이 분리되지 않도록 함)
  - 수치 보존(주가, 등락률, 매출액, 영업이익 등의 수치를 원문 그대로 유지)
  - `LangChain` 의 `RecursiveCharacterTextSplitter`
- 500~800자를 기준으로 100자 Overlap 적용
- 분할 기준	: 문단 및 문장 경계 우선
- 메타데이터 : 종목코드, 기업명, 기사 제목, 작성일, 언론사, URL
- 예외 처리 : 짧은 기사는 분할하지 않고 전체 유지

#### 3-3-2. 뉴스 데이터 인덱싱
- 데이터 수집/갱신 시 수행 (사용자 질의 시 해당 인덱스에서 검색 수행됨)
- FAISS
  - 임베딩(텍스트 → 벡터 변환) > 임베딩 벡터 > FAISS 인덱싱 > Vector Search (의미 기반)
  - Embedding Model을 통해 벡터화되어 Vector DB에 저장됨
    - `text-embedding-3-small` : 비용이 저렴하면서도 검색 정확도가 높음, FAISS 연산 부담이 무난한 범용 RAG 용도
- BM25
  - 텍스트 토큰화(키워드 처리) > BM25 인덱싱 > Keyword Search (키워드 기반)
  - 한국어 형태소 분석기(`Kiwi` 등)를 이용하여 뉴스 텍스트 토큰화

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

#### 4-3-1. Agent 기반 Query 분석
- 자연어 질문 > LLM 기반 분석 > 질문 구조화 (`ParsedQuestion`) > 작업 계획
- OpenAI LLM : Tool Calling 사용, 구조화된 JSON 출력을 위함
  - `gpt-5-mini` : 비용이 저렴하고 응답 속도가 빠름 / 질문 분류, 정보 추출, Tool Calling 등 반복 호출에 적합
``` bash
[질의] “삼전이 하이닉스보다 오르는 이유는?”

[작업 분해]
1. 두 종목의 동일 기준일·기간 결정
2. 종목 별칭, 질의 의도(상승 원인) 해석
3. 삼성전자/SK하이닉스 주가·추세 확보
4. 같은 기간의 상대적인 가격 흐름 비교
5. 두 기업의 뉴스·공시 검색
6. 비교 결과와 원인 후보 통합

.\.venv\Scripts\python.exe -X utf8 -c "import json; from src.parsers.stock_question_parser import parse_stock_question; print(json.dumps(parse_stock_question('유진로봇 최근 주가 전망?').to_dict(), ensure_ascii=False, indent=2))"
```

#### 4-3-2. Agent Controller
- 다음 작업 중에서 선택
  - 데이터 충분함 : 응답 생성
  - 데이터 부족 : 추가 필요한 작업만 선택하여 수집 절차 수행
  - 검색 표현 부적절 : 질문을 재작성하여 기존 인덱스 재검색
  - 한계 도달 : 부족한 근거를 명시하고 종료
- 데이터 수집 판단 기준
  - 주가 데이터 부족 : 시장 상태에 맞는 기준일, 완료 일봉 수, 추세 기준일 일치 여부
  - 뉴스 데이터 부족 : 질문의 종목·기간에 맞는 서로 다른 기사 수와 내용 (청크 수 X)
  - 뉴스 최신성 : 기사 발행일과 마지막 수집 시각을 함께 판단
  - 질문 충족 여부 : 상승 원인, 전망, 비교 등에 필요한 근거가 실제로 있는지 LLM이 평가
- 무한 반복과 비용 증가를 막도록 제한
  - 재시도 or 도구 호출 횟수
  - 질문 전체 Agent 결정 최대 3회
  - 종목별 뉴스 수집 최대 2페이지(60건), 동일 페이지 중복 호출 금지
  - 종목별 주가 추가 수집은 최대 1회
  - 새 기사 or 근거가 늘지 않거나 검색 결과가 반복되면 조기 종료
  - 최근 동일 조건으로 수집했지만 결과가 없었던 이력도 확인하여 반복 수집 방지
  - 인덱싱 호출, 처리 청크 수, 요청 시간에도 별도 상한을 둠

### 4-4. Retriever Layer
- 뉴스 데이터
  - 하이브리드 검색(Hybrid Search) 적용
    - BM25 알고리즘을 통한 정확한 키워드 일치 검색(Keyword Search) + 코사인 유사도(Cosine Similarity)를 활용한 FAISS 기반 Vector Search
    - Keyword Search : 질문의 핵심 키워드가 포함된 뉴스 검색 ("삼성전자", "주가", "상승" 등)
    - Vector Search : 질문과 의미적으로 유사한 뉴스 검색 ("주가 왜 올랐어?" <-> "매수세 증가", "계약 체결" 연결)
  - Hybrid Search를 통한 검색 결과 통합 및 재정렬
    - RRF 알고리즘을 통해 검색 결과 통합 + 상위 Top-K 뉴스를 반환
  - 데이터 부족 or 최신성 확인 필요 판단 -> 수집 실행
    - 수집 시, 최신 30건 데이터 조회 이후 Agent 판단에 따라 추가 조회 (`start=31`)
    - 데이터가 충분한 경우 기존 인덱스 연결, 수집 절차 생략함
  - 이후 `공시 문서, 기업 사업보고서` 등의 **텍스트 문서 데이터** 추가 예정
- 주가 데이터
  - 정형 데이터이므로 벡터 검색이 아닌 거래일 조건 기반 필터링 및 시계열 조회 방식

#### 4-4-1. Vector Search
- FAISS 인덱스에서 관련 뉴스 검색
  - `normalize_L2()` 후 내적하여 유사도 계산 -> 청크 정보와 score(코사인 유사도 값) 반환
  - 인덱스 검색 방식 타입별 응답 시간 테스트 결과 -> `Flat` 사용

#### 4-4-2. Keyword Search
- `Kiwi` : 한국어 문장을 형태소 단위로 나누는 분석기
  - 입력 : "삼성전자 실적이 증가했다"
  - 분석 : 삼성전자 / 실적 / 이 / 증가 / 하 / 었 / 다

#### 4-4-3. Hybrid Search
- Vector Search + Keyword Search 실행
- `chunk_id` 기준으로 중복을 합치고, **가중 점수 합산(Weighted Sum)**로 두 검색 순위를 결합
  - 가중 점수 합산(Weighted Sum) : 각 검색 점수 정규화 후 가중치를 적용해 합산

#### 4-4-4. Reranking
- 검색된 뉴스 중 `important="O"` 조건의 데이터를 우선함 (원래 `score`는 유지)
  - 금융/경제 뉴스를 우선해야 하기 때문임
- 같은 중요도에서는 Hybrid Search 순위(`hybrid_score`) 유지
- 최종 `top_k` 선택

### 4-5. Context Builder
- LLM 입력용 컨텍스트 구성
- 주가/뉴스 데이터를 LLM 입력에 적합한 순서로 정렬하고 재구성
  - *향후 고도화 방안 : 지식 그래프 구조를 활용하여 데이터 간 관계를 정렬하고, 기준에 따라 순서를 재배열(Re-ranking)*
- 불필요한 정보는 제거하고 핵심 정보만 유지 -> LLM이 정확한 추론을 수행하도록 함

### 4-6. LLM Generator
- 구성된 컨텍스트를 기반으로 뉴스와 주가 간의 관계를 분석하고 자연어 응답을 생성
- 단순 요약이 아닌 원인(뉴스/공시/리포트 등) → 결과(주가) 구조
- 원본 검색 데이터와 대조하는 검증 절차를 포함해 Hallucination을 최소화하고 신뢰성을 확보
- 뉴스
  - 감성 분석(positive/neutral/negative) 실시 -> 사용자 판단에 도움이 되는 지표 생성

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

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
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data root
DATA_DIR = PROJECT_ROOT / "data"

# Sample data
SAMPLE_DATA_DIR = DATA_DIR / "sample"
SAMPLE_STOCKS_PATH = SAMPLE_DATA_DIR / "sample_stocks.json"
SAMPLE_NEWS_PATH = SAMPLE_DATA_DIR / "sample_news.json"

# Reference data
REFERENCE_DATA_DIR = DATA_DIR / "reference"
STOCK_NAME_CODE_PATH = REFERENCE_DATA_DIR / "stock_name_code.json"

# Raw data
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PRICES_DIR = RAW_DATA_DIR / "prices"
RAW_NEWS_DIR = RAW_DATA_DIR / "news"
RAW_REPORTS_DIR = RAW_DATA_DIR / "reports"
RAW_DISCLOSURES_DIR = RAW_DATA_DIR / "disclosures"
RAW_FINANCIALS_DIR = RAW_DATA_DIR / "financials"

# Processed data
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_PRICES_DIR = PROCESSED_DATA_DIR / "prices"
PROCESSED_NEWS_DIR = PROCESSED_DATA_DIR / "news"
PROCESSED_REPORTS_DIR = PROCESSED_DATA_DIR / "reports"
PROCESSED_DISCLOSURES_DIR = PROCESSED_DATA_DIR / "disclosures"
PROCESSED_FINANCIALS_DIR = PROCESSED_DATA_DIR / "financials"

# Chunked document data
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
NEWS_CHUNKS_DIR = CHUNKS_DATA_DIR / "news"
REPORTS_CHUNKS_DIR = CHUNKS_DATA_DIR / "reports"
DISCLOSURES_CHUNKS_DIR = CHUNKS_DATA_DIR / "disclosures"
FINANCIALS_CHUNKS_DIR = CHUNKS_DATA_DIR / "financials"

# FAISS vector store
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FAISS_VECTOR_STORE_DIR = VECTOR_STORE_DIR / "faiss"
FAISS_INDEX_PATH = FAISS_VECTOR_STORE_DIR / "stock_advisor.index"
FAISS_METADATA_PATH = FAISS_VECTOR_STORE_DIR / "stock_advisor_metadata.json"


# Stock price file paths
def get_raw_price_path(stock_code: str) -> Path:
    return RAW_PRICES_DIR / f"{stock_code}_daily.json"


def get_processed_price_trend_path(stock_code: str) -> Path:
    return PROCESSED_PRICES_DIR / f"{stock_code}_trend.json"


# Chunked document file paths
def get_news_chunks_path(stock_code: str) -> Path:
    return NEWS_CHUNKS_DIR / f"{stock_code}_news_chunks.json"


def get_reports_chunks_path(stock_code: str) -> Path:
    return REPORTS_CHUNKS_DIR / f"{stock_code}_reports_chunks.json"


def get_disclosures_chunks_path(stock_code: str) -> Path:
    return DISCLOSURES_CHUNKS_DIR / f"{stock_code}_disclosures_chunks.json"


def get_financials_chunks_path(stock_code: str) -> Path:
    return FINANCIALS_CHUNKS_DIR / f"{stock_code}_financials_chunks.json"

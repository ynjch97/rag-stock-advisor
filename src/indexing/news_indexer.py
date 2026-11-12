from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from kiwipiepy import Kiwi
from openai import OpenAI

from src.config.paths import BM25_TOKENS_PATH, NEWS_CHUNKS_DIR, PROJECT_ROOT, RAW_NEWS_DIR
from src.config.settings import Settings, settings
from src.collectors.news_collector import summarize_news_page
from src.domain.news_models import NewsChunk
from src.embeddings.embedding_service import build_news_index_texts, embed_texts
from src.indexing.bm25_index import (
    BM25_PARAMETERS, TOKENIZER_SPEC, build_bm25_index, load_bm25_index, save_bm25_index,
)
from src.indexing.faiss_index import build_faiss_index, load_faiss_index, save_faiss_index
from src.services.execution_budget import remaining_timeout


# 청킹 파일에 대응하는 원본 수집 시각·기간·페이지를 읽어 성공한 인덱싱 이력 구성
def _load_news_collections(chunks_dir: Path, raw_news_dir: Path) -> list[dict]:
    history = []
    for path in sorted(chunks_dir.glob("*_chunks.json")):
        raw_path = raw_news_dir / (path.name[:-len("_chunks.json")] + ".json")
        if not raw_path.exists():
            continue
        with raw_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
            collection = raw.get("collection", {})
        if not all(collection.get(key) for key in ("stock_code", "start_date", "end_date", "collected_at")):
            continue
        collection = {**collection, **summarize_news_page(
            raw, collection["start_date"], collection["end_date"], collection.get("start", 1),
        )}
        collected_at = datetime.fromisoformat(collection["collected_at"])
        if collected_at.utcoffset() is None:
            raise ValueError("News collection time requires a timezone.")
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        urls = {record["url"] for record in records
                if collection["start_date"] <= record["published_date"] <= collection["end_date"]}
        history.append({**collection, "article_count": len(urls), "chunk_file": path.name})
    return sorted(history, key=lambda item: item["collected_at"])[-500:]


# 빈 수집 결과·변경 없는 인덱스도 수집 이력을 원자적으로 갱신
def _save_index_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# 기사별 최신 스냅샷 전체를 선택하고 두 인덱스가 공유할 청크 순서 확정
def load_latest_news_chunks(chunks_dir: Path = NEWS_CHUNKS_DIR) -> list[NewsChunk]:
    articles: dict[tuple[str, str], list[NewsChunk]] = {}
    for path in sorted(chunks_dir.glob("*.json"), key=_snapshot_order):
        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"News chunks must contain a JSON array: {path.name}")
        groups: dict[tuple[str, str], list[NewsChunk]] = {}
        for record in records:
            try:
                chunk = NewsChunk(**record)
            except TypeError:
                raise ValueError(f"Invalid news chunk record: {path.name}") from None
            _validate_chunk(chunk)
            groups.setdefault((chunk.stock_code, chunk.url), []).append(chunk)
        for key, group in groups.items():
            group.sort(key=lambda chunk: chunk.chunk_index)
            _validate_article(group)
            articles[key] = group
    chunks = [chunk for key in sorted(articles) for chunk in articles[key]]
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Different articles have colliding chunk IDs; regenerate those chunk snapshots.")
    return chunks


# 수집 시각이 있는 파일명은 그 시각을 우선하고 사용자 지정 이름은 수정 시각 사용
def _snapshot_order(path: Path) -> tuple[float, str]:
    match = re.search(r"_(\d{8}_\d{6}_\d{6})_chunks\.json$", path.name)
    timestamp = (datetime.strptime(match[1], "%Y%m%d_%H%M%S_%f")
                 .replace(tzinfo=timezone(timedelta(hours=9))).timestamp() if match else path.stat().st_mtime)
    return timestamp, path.name


# 식별자·본문·메타데이터와 청크 위치 필드 검증
def _validate_chunk(chunk: NewsChunk) -> None:
    if (any(not isinstance(value, str) or not value.strip() for value in (
        chunk.chunk_id, chunk.news_id, chunk.stock_name, chunk.stock_code, chunk.url,
        chunk.title, chunk.content, chunk.publisher, chunk.published_date,
    )) or chunk.important not in ("O", "X")
            or type(chunk.chunk_index) is not int or chunk.chunk_index < 1
            or type(chunk.chunk_count) is not int or chunk.chunk_count < 1
            or type(chunk.start_index) is not int or chunk.start_index < 0):
        raise ValueError("Invalid news chunk fields.")
    date.fromisoformat(chunk.published_date)


# 서로 다른 버전의 기사 청크가 섞이거나 일부가 누락된 경우 인덱싱 중단
def _validate_article(chunks: list[NewsChunk]) -> None:
    metadata = {(chunk.news_id, chunk.stock_name, chunk.title, chunk.important,
                 chunk.publisher, chunk.published_date, chunk.chunk_count) for chunk in chunks}
    if (len(metadata) != 1 or chunks[0].chunk_count != len(chunks)
            or [chunk.chunk_index for chunk in chunks] != list(range(1, len(chunks) + 1))
            or len({chunk.chunk_id for chunk in chunks}) != len(chunks)):
        raise ValueError("Incomplete or mixed-version news article chunks.")


# FAISS·BM25·메타데이터의 체크섬과 문서 순서·개수·차원을 검증하여 재로딩
def load_news_indexes(faiss_dir: Path, bm25_path: Path = BM25_TOKENS_PATH) -> tuple:
    with (faiss_dir / "stock_advisor_metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("chunks"), list):
        raise ValueError("Invalid news index metadata.")
    index_path = faiss_dir / "stock_advisor.index"
    if (_file_hash(index_path) != metadata.get("faiss_sha256")
            or _file_hash(bm25_path) != metadata.get("bm25_sha256")):
        raise ValueError("News index files do not match; rebuild with force=True.")
    index = load_faiss_index(index_path)
    bm25, corpus = load_bm25_index(bm25_path)
    try:
        chunks = [NewsChunk(**record) for record in metadata["chunks"]]
    except TypeError:
        raise ValueError("Invalid saved news chunk metadata.") from None
    for chunk in chunks:
        _validate_chunk(chunk)
    ids = [chunk.chunk_id for chunk in chunks]
    if (len(set(ids)) != len(ids) or index.ntotal != len(chunks)
            or index.d != metadata.get("dimension") or len(chunks) != metadata.get("chunk_count")
            or corpus["chunk_ids"] != ids or corpus.get("fingerprint") != metadata.get("fingerprint")):
        raise ValueError("FAISS, BM25, and metadata refer to different chunk lists.")
    return index, bm25, metadata


# 청크 변경 시에만 전체 재구축; 모든 산출물 검증 후 기존 파일을 교체
def index_news(
    chunks_dir: Path = NEWS_CHUNKS_DIR, *, app_settings: Settings = settings,
    faiss_dir: Path | None = None, bm25_path: Path = BM25_TOKENS_PATH,
    client: OpenAI | None = None, kiwi: Kiwi | None = None,
    batch_size: int = 32, force: bool = False,
    raw_news_dir: Path = RAW_NEWS_DIR, max_chunks: int | None = None,
    max_characters: int | None = None, deadline: float | None = None,
) -> dict:
    remaining_timeout(deadline)
    if app_settings.vector_db_provider.lower() != "faiss":
        raise ValueError("News indexing currently requires VECTOR_DB_PROVIDER=faiss.")
    chunks = load_latest_news_chunks(chunks_dir)
    collections = _load_news_collections(chunks_dir, raw_news_dir)
    destination = faiss_dir if faiss_dir is not None else PROJECT_ROOT / app_settings.vector_db_path
    index_path, metadata_path = destination / "stock_advisor.index", destination / "stock_advisor_metadata.json"
    if not chunks:
        if not collections or index_path.exists():
            raise ValueError("No news chunks to index.")
        _save_index_metadata(metadata_path, {"schema_version": 1, "chunks": [], "chunk_count": 0,
                                           "collections": collections, "dimension": 0})
        return {"rebuilt": False, "chunk_count": 0, "dimension": 0,
                "faiss_path": str(index_path), "bm25_path": str(bm25_path)}
    if max_chunks is not None and len(chunks) > max_chunks:
        raise ValueError("News indexing chunk limit exceeded.")
    if len({path.resolve() for path in (index_path, metadata_path, bm25_path)}) != 3:
        raise ValueError("Index artifacts must use three separate file paths.")
    config = {"embedding_model": app_settings.openai_embedding_model, "input_format": "stock-title-content-v1",
              "metric": "cosine", "tokenizer": TOKENIZER_SPEC, "bm25_parameters": BM25_PARAMETERS}
    fingerprint = hashlib.sha256(json.dumps(
        {"chunks": [asdict(chunk) for chunk in chunks], "config": config},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    if not force and all(path.exists() for path in (index_path, metadata_path, bm25_path)):
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, dict):
            raise ValueError("Invalid news index metadata; rebuild with force=True.")
        if existing.get("fingerprint") == fingerprint:
            _, _, existing = load_news_indexes(destination, bm25_path)
            if existing.get("collections", []) != collections:
                _save_index_metadata(metadata_path, {**existing, "collections": collections})
            return {"rebuilt": False, "chunk_count": len(chunks), "dimension": existing["dimension"],
                    "faiss_path": str(index_path), "bm25_path": str(bm25_path)}

    texts = build_news_index_texts(chunks)
    if max_characters is not None and sum(map(len, texts)) > max_characters:
        raise ValueError("News indexing character limit exceeded.")
    # 토큰화 실패도 유료 임베딩 요청 전에 확인
    _, tokens = build_bm25_index(texts, kiwi=kiwi)
    options = {"deadline": deadline} if deadline is not None else {}
    remaining_timeout(deadline)
    vectors = embed_texts(texts, app_settings=app_settings, client=client, batch_size=batch_size, **options)
    index = build_faiss_index(vectors)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".news_index_", dir=destination.parent) as directory:
        staging = Path(directory)
        staged_index, staged_bm25 = staging / index_path.name, staging / "tokens.json"
        staged_metadata = staging / metadata_path.name
        save_faiss_index(index, staged_index)
        save_bm25_index(tokens, [chunk.chunk_id for chunk in chunks], staged_bm25, fingerprint=fingerprint)
        metadata = {"schema_version": 1, "fingerprint": fingerprint, "configuration": config,
                    "embedding_model": app_settings.openai_embedding_model,
                    "dimension": index.d, "chunk_count": len(chunks),
                    "chunks": [asdict(chunk) for chunk in chunks],
                    "collections": collections,
                    "faiss_sha256": _file_hash(staged_index), "bm25_sha256": _file_hash(staged_bm25)}
        with staged_metadata.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
        load_news_indexes(staging, staged_bm25)
        remaining_timeout(deadline)
        _publish_files([(staged_index, index_path), (staged_bm25, bm25_path), (staged_metadata, metadata_path)])
    return {"rebuilt": True, "chunk_count": len(chunks), "dimension": index.d,
            "faiss_path": str(index_path), "bm25_path": str(bm25_path)}


# 벡터와 토큰 파일의 변경·불일치를 메타데이터로 확인하기 위한 체크섬 계산
def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 교체 전 기존 파일을 백업하고 중간 교체 실패 시 이미 바뀐 파일도 복구
def _publish_files(files: list[tuple[Path, Path]]) -> None:
    backups = []
    for staged, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = staged.with_name(staged.name + ".backup") if target.exists() else None
        if backup is not None:
            shutil.copy2(target, backup)
        backups.append((staged, target, backup))
    replaced = []
    try:
        for staged, target, backup in backups:
            staged.replace(target)
            replaced.append((target, backup))
    except BaseException:
        for target, backup in reversed(replaced):
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                backup.replace(target)
        raise

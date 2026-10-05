from functools import lru_cache

import chromadb

from util.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _embedding_function():
    """Reuse the ingest embedding model for queries."""
    from agents.shared.embedding_models import get_chroma_embedding_function

    return get_chroma_embedding_function()


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path="data/chroma")


def get_or_create_collection(ticker: str) -> chromadb.Collection:
    client = get_chroma_client()
    name = f"filings_{ticker.lower()}"
    embedding_fn = _embedding_function()
    try:
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        logger.warning(f"Reopening filings collection {name}: {exc}")
        return client.get_collection(name, embedding_function=embedding_fn)


def collection_exists(ticker: str) -> bool:
    client = get_chroma_client()
    collection_name = f"filings_{ticker.lower()}"

    try:
        collections = client.list_collections()
        return any(c.name == collection_name for c in collections)
    except Exception:
        return False


def get_collection_stats(ticker: str) -> dict:
    if not collection_exists(ticker):
        return {"exists": False, "document_count": 0, "latest_filing_date": None}

    collection = get_or_create_collection(ticker)
    count = collection.count()
    latest_date = None

    if count:
        results = collection.get(include=["metadatas"])
        dates = [
            meta.get("filing_date")
            for meta in results.get("metadatas") or []
            if meta and meta.get("filing_date")
        ]
        if dates:
            latest_date = max(dates)

    if latest_date is None:
        metadata = collection.metadata or {}
        latest_date = metadata.get("latest_filing_date") or None

    return {
        "exists": True,
        "document_count": count,
        "latest_filing_date": latest_date,
    }


def record_latest_filing_date(ticker: str, latest_filing_date: str) -> None:
    """Store the newest filing date on the collection so freshness survives restarts."""
    collection = get_or_create_collection(ticker)
    metadata = dict(collection.metadata or {})
    metadata["latest_filing_date"] = latest_filing_date
    collection.modify(metadata=metadata)


def delete_collection(ticker: str) -> bool:
    client = get_chroma_client()
    collection_name = f"filings_{ticker.lower()}"

    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False

import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agents.shared.embedding_models import EMBEDDING_MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path="data/chroma")


def get_or_create_collection(ticker: str) -> chromadb.Collection:
    client = get_chroma_client()

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODELS["hf_embed_fast"],
        device="cpu",
        normalize_embeddings=True,
    )

    return client.get_or_create_collection(
        name=f"filings_{ticker.lower()}",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

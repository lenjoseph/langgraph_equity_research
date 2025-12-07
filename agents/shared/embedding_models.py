from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODELS = {
    "hf_embed_fast": "all-MiniLM-L6-v2",
    "hf_embed_balanced": "bge-small-en-v1.5",
    "hf_embed_high_dims": "bge-base-en-v1.5",
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    get cached hf embeddings instance
    first call downloads model (~130MB)
    """
    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODELS["hf_embed_fast"],
        model_kwargs={"device": "cpu"},
        # remove vector length bias from similarity search
        encode_kwargs={"normalize_embeddings": True},
    )

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
import torch

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

    torch.set_num_threads(2)

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODELS["hf_embed_fast"],
        model_kwargs={"device": "cpu"},
        # remove vector length bias from similarity search
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_chroma_embedding_function():
    """Chroma query embedder backed by the same model used during ingest."""
    from chromadb import EmbeddingFunction

    class SharedMiniLMEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):
            return get_embeddings().embed_documents(list(input))

        @staticmethod
        def name() -> str:
            return "shared_minilm"

        def get_config(self) -> dict:
            return {"model": EMBEDDING_MODELS["hf_embed_fast"]}

        @staticmethod
        def build_from_config(config):
            return SharedMiniLMEmbeddingFunction()

    return SharedMiniLMEmbeddingFunction()

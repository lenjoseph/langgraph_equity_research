from chromadb import Collection
from models.agent import FilingChunk


def embed_chunks(chunks: list[FilingChunk], collection: Collection) -> None:
    from agents.shared.embedding_models import get_embeddings

    embeddings_model = get_embeddings()
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts = [chunk.text for chunk in batch]
        ids = [f"{chunk.accession_number}_{chunk.chunk_index}" for chunk in batch]
        metadatas = [chunk.model_dump(exclude={"text"}) for chunk in batch]

        embeddings = embeddings_model.embed_documents(texts)

        collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

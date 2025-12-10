"""Chunk parsed filings into embeddable segments."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from util.logger import get_logger
from models.agent import FilingChunk, FilingMetadata

logger = get_logger(__name__)

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len,
)


def chunk_filing(
    parsed_sections: dict[str, str],
    metadata: FilingMetadata,
) -> list[FilingChunk]:
    """
    Chunk a parsed filing into embeddable segments.

    Args:
        parsed_sections: Dict mapping section names to text content
        metadata: Filing metadata

    Returns:
        List of FilingChunk objects
    """
    chunks = []
    chunk_index = 0

    for section_name, section_text in parsed_sections.items():
        if not section_text or len(section_text.strip()) < 50:
            continue

        text_chunks = splitter.split_text(section_text)

        for text in text_chunks:
            chunks.append(
                FilingChunk(
                    text=text,
                    ticker=metadata.ticker,
                    filing_type=metadata.filing_type,
                    section=section_name,
                    filing_date=metadata.filing_date,
                    accession_number=metadata.accession_number,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    logger.info(
        f"Created {len(chunks)} chunks from {metadata.filing_type} "
        f"({metadata.accession_number})"
    )
    return chunks

"""Parse SEC filing HTML into structured sections."""

import re
from typing import Optional
from bs4 import BeautifulSoup

from util.logger import get_logger

logger = get_logger(__name__)

SECTION_PATTERNS = {
    "10-K": {
        "business": r"item\s*1[.\s]+business",
        "risk_factors": r"item\s*1a[.\s]+risk\s*factors",
        "mda": r"item\s*7[.\s]+management",
        "financial_statements": r"item\s*8[.\s]+financial\s*statements",
    },
    "10-Q": {
        "financial_statements": r"part\s*i[.\s]+financial\s*information",
        "mda": r"item\s*2[.\s]+management",
        "risk_factors": r"item\s*1a[.\s]+risk\s*factors",
    },
    "8-K": {
        "current_report": r"item\s*\d+\.\d+",
    },
}


def clean_text(text: str) -> str:
    """Clean extracted text by removing excess whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML content."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    return clean_text(text)


def find_section_boundaries(
    text: str, patterns: dict[str, str]
) -> dict[str, tuple[int, int]]:
    """
    Find start/end positions of sections in text.

    Returns dict mapping section name to (start, end) positions.
    """
    boundaries = {}
    section_starts = []

    for section_name, pattern in patterns.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            start = matches[0].start()
            section_starts.append((start, section_name))

    section_starts.sort(key=lambda x: x[0])

    for i, (start, name) in enumerate(section_starts):
        if i + 1 < len(section_starts):
            end = section_starts[i + 1][0]
        else:
            end = len(text)
        boundaries[name] = (start, end)

    return boundaries


def parse_10k(raw_html: str) -> dict[str, str]:
    """
    Parse 10-K filing into sections.

    Returns:
        Dict mapping section names to extracted text
    """
    text = extract_text_from_html(raw_html)
    patterns = SECTION_PATTERNS["10-K"]
    boundaries = find_section_boundaries(text, patterns)

    sections = {}
    for section_name, (start, end) in boundaries.items():
        section_text = text[start:end]
        if len(section_text) > 100:  # Skip very short sections
            sections[section_name] = section_text

    if not sections:
        logger.warning("Could not parse 10-K sections, using full text")
        sections["full_document"] = text

    return sections


def parse_10q(raw_html: str) -> dict[str, str]:
    """
    Parse 10-Q filing into sections.

    Returns:
        Dict mapping section names to extracted text
    """
    text = extract_text_from_html(raw_html)
    patterns = SECTION_PATTERNS["10-Q"]
    boundaries = find_section_boundaries(text, patterns)

    sections = {}
    for section_name, (start, end) in boundaries.items():
        section_text = text[start:end]
        if len(section_text) > 100:
            sections[section_name] = section_text

    if not sections:
        logger.warning("Could not parse 10-Q sections, using full text")
        sections["full_document"] = text

    return sections


def parse_8k(raw_html: str) -> dict[str, str]:
    """
    Parse 8-K filing (material events).

    Returns:
        Dict with 'current_report' containing the full text
    """
    text = extract_text_from_html(raw_html)
    return {"current_report": text}


def parse_filing(raw_html: str, filing_type: str) -> dict[str, str]:
    """
    Parse a filing based on its type.

    Args:
        raw_html: Raw HTML content
        filing_type: Type of filing (10-K, 10-Q, 8-K)

    Returns:
        Dict mapping section names to extracted text
    """
    parsers = {
        "10-K": parse_10k,
        "10-Q": parse_10q,
        "8-K": parse_8k,
    }

    parser = parsers.get(filing_type)
    if parser:
        return parser(raw_html)

    logger.warning(f"Unknown filing type: {filing_type}, extracting full text")
    return {"full_document": extract_text_from_html(raw_html)}

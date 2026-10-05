"""Deterministic checks for the shared synthesis template."""

import re

from agents.aggregation.rubric import MAX_SYNTHESIS_WORDS, REQUIRED_SECTIONS

_SENTIMENT_RE = re.compile(
    r"overall sentiment:\**\s*(bullish|bearish|neutral)\b",
    re.IGNORECASE,
)


def check_aggregation_format(text: str | None) -> list[str]:
    """Return format violations. An empty list means the template is satisfied."""
    if not text or not text.strip():
        return ["Synthesis is empty."]

    issues: list[str] = []
    word_count = len(text.split())
    if word_count > MAX_SYNTHESIS_WORDS:
        issues.append(
            f"Synthesis is {word_count} words; the limit is {MAX_SYNTHESIS_WORDS}."
        )

    lowered = text.lower()
    for section in REQUIRED_SECTIONS:
        if section.lower() not in lowered:
            issues.append(f"Missing section: {section.rstrip(':')}.")

    if not _SENTIMENT_RE.search(text):
        issues.append("Overall Sentiment must be BULLISH, BEARISH, or NEUTRAL.")
    return issues

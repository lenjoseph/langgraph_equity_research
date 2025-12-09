from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# ============================================================================
# Shared Enums
# ============================================================================


class Confidence(str, Enum):
    """Confidence level for sentiment analysis."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Sentiment(str, Enum):
    """Directional sentiment (BULLISH/BEARISH/NEUTRAL)."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RelativeSentiment(str, Enum):
    """Relative sentiment (POSITIVE/NEGATIVE/NEUTRAL) for comparisons."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class Valuation(str, Enum):
    """Valuation assessment for fundamental analysis."""

    UNDERVALUED = "UNDERVALUED"
    OVERVALUED = "OVERVALUED"
    FAIRLY_VALUED = "FAIRLY VALUED"


# ============================================================================
# Shared Models
# ============================================================================


class KeyPointWithCitation(BaseModel):
    """A key point with source citation (for web-grounded research)."""

    point: str = Field(description="The key point or insight from the research")
    source: str = Field(description="The source/publication name")
    date: str = Field(description="The date of the source")


# ============================================================================
# Agent Output Models
# ============================================================================


class HeadlineSentimentOutput(BaseModel):
    """Structured output for headline sentiment analysis."""

    sentiment: Sentiment = Field(
        description="Overall sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    key_points: List[KeyPointWithCitation] = Field(
        description="List of 2-3 key supporting points with citations"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


class FundamentalSentimentOutput(BaseModel):
    """Structured output for fundamental sentiment analysis."""

    valuation: Valuation = Field(
        description="Overall valuation: UNDERVALUED, OVERVALUED, or FAIRLY VALUED"
    )
    key_points: List[str] = Field(
        description="List of 2-3 key supporting points from financial metrics"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


class TechnicalSentimentOutput(BaseModel):
    """Structured output for technical sentiment analysis."""

    sentiment: Sentiment = Field(
        description="Overall technical sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    key_points: List[str] = Field(
        description="List of up to 5 key supporting points referencing specific indicators"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


class MacroSentimentOutput(BaseModel):
    """Structured output for macro sentiment analysis."""

    sentiment: Sentiment = Field(
        description="Overall macro sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    key_points: List[str] = Field(description="List of 2-3 key supporting points")
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


class IndustrySentimentOutput(BaseModel):
    """Structured output for industry sentiment analysis."""

    sentiment: RelativeSentiment = Field(
        description="Overall industry sentiment: POSITIVE, NEGATIVE, or NEUTRAL"
    )
    key_points: List[KeyPointWithCitation] = Field(
        description="List of 2-3 key supporting points with citations"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


class PeerSentimentOutput(BaseModel):
    """Structured output for peer sentiment analysis."""

    sentiment: RelativeSentiment = Field(
        description="Overall peer sentiment: POSITIVE, NEGATIVE, or NEUTRAL (relative to peers)"
    )
    key_points: List[KeyPointWithCitation] = Field(
        description="List of 2-3 key supporting points with citations"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")


# ============================================================================
# Evaluator Models
# ============================================================================


class AggregatorFeedback(BaseModel):
    """Feedback from the sentiment evaluator."""

    compliant: bool = Field(
        description="Determine if the provided sentiment complies with the output structure"
    )
    feedback: str = Field(
        description="If the output is noncompliant, provide feedback on how to address the delta. Otherwise do not include feedback"
    )


# ============================================================================
# SEC Filing Models
# ============================================================================


class FilingMetadata(BaseModel):
    ticker: str
    filing_type: str
    filing_date: str
    accession_number: str
    url: str


class FilingChunk(BaseModel):
    text: str
    ticker: str
    filing_type: str
    section: str
    filing_date: str
    accession_number: str
    chunk_index: int


class FilingCitation(BaseModel):
    """Citation from an SEC filing."""

    quote: str = Field(description="Direct quote or paraphrase from the filing")
    filing_type: str = Field(description="Type of filing (10-K, 10-Q, 8-K)")
    section: str = Field(description="Section of the filing (e.g., risk_factors, mda)")
    filing_date: str = Field(description="Date of the filing (YYYY-MM-DD)")


class FilingsSentimentOutput(BaseModel):
    """Structured output for SEC filings sentiment analysis."""

    sentiment: Sentiment = Field(
        description="Overall sentiment: BULLISH, BEARISH, or NEUTRAL"
    )
    key_findings: List[str] = Field(
        description="List of 3-5 key findings from the filings"
    )
    citations: List[FilingCitation] = Field(
        description="Citations supporting the key findings"
    )
    risk_factors_summary: str = Field(
        description="Summary of material risk factors identified"
    )
    confidence: Confidence = Field(description="Confidence level: High, Medium, or Low")

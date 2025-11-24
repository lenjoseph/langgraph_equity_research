from pydantic import BaseModel, Field


class FundamentalAnalysisOutput(BaseModel):
    """Structured output model for fundamental analysis."""

    sentiment: str = Field(
        description="Overall fundamental sentiment: 'undervalued', 'overvalued', or 'fairly valued'"
    )
    key_points: list[str] = Field(
        description="Exactly 3 key supporting points from financial metrics. You must provide exactly 3 points.",
        min_length=3,
        max_length=3,
    )
    confidence: str = Field(
        description="Confidence level in the analysis: 'high', 'medium', or 'low'"
    )


class TechnicalAnalysisOutput(BaseModel):
    """Structured output model for technical analysis."""

    sentiment: str = Field(
        description="Overall technical sentiment: 'bullish', 'bearish', or 'neutral'"
    )
    key_points: list[str] = Field(
        description="Exactly 3 key supporting points from technical indicators. You must provide exactly 3 points.",
        min_length=3,
        max_length=3,
    )
    confidence: str = Field(
        description="Confidence level in the analysis: 'high', 'medium', or 'low'"
    )


class MacroAnalysisOutput(BaseModel):
    """Structured output model for macro analysis."""

    sentiment: str = Field(
        description="Overall macro sentiment: 'favorable', 'unfavorable', or 'neutral'"
    )
    key_points: list[str] = Field(
        description="Exactly 3 key supporting points from macro indicators. You must provide exactly 3 points.",
        min_length=3,
        max_length=3,
    )
    confidence: str = Field(
        description="Confidence level in the analysis: 'high', 'medium', or 'low'"
    )


class IndustryAnalysisOutput(BaseModel):
    """Structured output model for industry analysis."""

    sentiment: str = Field(
        description="Overall industry sentiment: 'positive', 'negative', or 'neutral'"
    )
    key_points: list[str] = Field(
        description="Exactly 3 key supporting points from industry trends. You must provide exactly 3 points.",
        min_length=3,
        max_length=3,
    )
    confidence: str = Field(
        description="Confidence level in the analysis: 'high', 'medium', or 'low'"
    )


class HeadlineAnalysisOutput(BaseModel):
    """Structured output model for headline analysis."""

    sentiment: str = Field(
        description="Overall headline sentiment: 'positive', 'negative', or 'neutral'"
    )
    key_points: list[str] = Field(
        description="Exactly 3 key supporting points from recent news and headlines. You must provide exactly 3 points.",
        min_length=3,
        max_length=3,
    )
    confidence: str = Field(
        description="Confidence level in the analysis: 'high', 'medium', or 'low'"
    )

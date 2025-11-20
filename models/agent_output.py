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


class ResearchSummaries(BaseModel):
    """Nested model for research summaries."""

    fundamental: str = Field(description="Summary of fundamental analysis takeaways")
    technical: str = Field(description="Summary of technical analysis takeaways")
    macro: str = Field(description="Summary of macro analysis takeaways")
    industry: str = Field(description="Summary of industry analysis takeaways")
    headline: str = Field(description="Summary of headline analysis takeaways")


class AggregatedSentimentOutput(BaseModel):
    """Structured output model for aggregated sentiment analysis."""

    trade_duration_category: str = Field(
        description="Trade duration category: 'Short-term', 'Medium-term', or 'Long-term'"
    )
    research_summaries: ResearchSummaries = Field(
        description="Summaries of each research agent's findings"
    )
    overall_sentiment: str = Field(
        description="Overall investment sentiment: 'BULLISH', 'BEARISH', or 'NEUTRAL'"
    )
    conclusion: str = Field(
        description="3-4 sentence conclusion synthesizing the most important factors"
    )

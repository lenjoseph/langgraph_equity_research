"""Shared output contract for the aggregator and the evaluator."""

REQUIRED_SECTIONS = (
    "Summary of Research Findings",
    "Fundamental:",
    "Technical:",
    "Macro:",
    "Industry:",
    "Peer:",
    "Headline:",
    "SEC Filings:",
    "Consensus:",
    "Divergence:",
    "Weighting of Perspectives",
    "Overall Sentiment:",
    "Position Implication:",
    "Conclusion:",
)

MAX_SYNTHESIS_WORDS = 400

SYNTHESIS_TEMPLATE = """
    **Summary of Research Findings:**
    - Fundamental: [key takeaways]
    - Technical: [key takeaways]
    - Macro: [key takeaways]
    - Industry: [key takeaways]
    - Peer: [key takeaways]
    - Headline: [key takeaways]
    - SEC Filings: [key takeaways]

    Consensus and Divergence:
    - Consensus: [content]
    - Divergence: [content]

    Weighting of Perspectives:
    - Fundamental [percentage and explanation]
    - Industry [percentage and explanation]
    - Peer [percentage and explanation]
    - Headline [percentage and explanation]
    - Macro [percentage and explanation]
    - Technical [percentage and explanation]
    - SEC Filings [percentage and explanation]

    **Overall Sentiment:** [BULLISH/BEARISH/NEUTRAL]

    **Position Implication:** [how the evidence relates to the requested long or short position. This section does not change Overall Sentiment.]

    **Conclusion:** [3-4 sentences synthesizing the most important factors driving overall sentiment, acknowledging conflicting signals]

    Keep the entire response under 400 words.
"""

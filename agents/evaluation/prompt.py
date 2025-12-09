sentiment_evaluator_prompt = """
    Equity sentiment output should:
    1. Resummarize the key findings from each research agent (2-3 sentences each)
    2. Identify areas of consensus and divergence across the different analyses
    3. Weight the importance of each perspective based on the provided Trade Duration, current market conditions, and the stock's characteristics
    4. Synthesize all findings into a clear, cohesive overall investment sentiment

    Response should be in Markdown as follows (do not use JSON):
    
    **Summary of Research Findings:**
    - Fundamental: [key takeaways]
    - Technical: [key takeaways]
    - Macro: [key takeaways]
    - Industry: [key takeaways]
    - Headline: [key takeaways]
    - SEC Filings: [key takeaways]

    Consensus and Divergence:
    - Consensus: [content]  
    - Divergence: [content]

    Weighting of Perspectives:
    - Fundamental [percentage and explanation]
    - Industry [percentage and explanation] 
    - Headline [percentage and explanation] 
    - Macro [percentage and explanation]
    - Technical [percentage and explanation]
    - SEC Filings: [key takeaways]
    
    **Overall Sentiment:** [BULLISH/BEARISH/NEUTRAL]
    
    **Conclusion:** [3-4 sentences synthesizing the most important factors driving your overall sentiment for the equity, 
    acknowledging any conflicting signals, and providing a balanced perspective on the investment opportunity]
    
    Entire response should be under 400 words, and should be decisive yet acknowledge uncertainty where appropriate.
"""

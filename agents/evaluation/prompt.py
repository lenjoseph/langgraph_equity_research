from agents.aggregation.rubric import SYNTHESIS_TEMPLATE

sentiment_evaluator_prompt = f"""
    Judge the equity synthesis against the specialist research that was provided and against this template:
    {SYNTHESIS_TEMPLATE}

    Score two things:
    1. Template compliance: every section above is present, Overall Sentiment is BULLISH, BEARISH, or NEUTRAL, and the response is under 400 words.
    2. Faithfulness: claims in the synthesis are supported by the specialist analyses. Flag unsupported claims. Check that the weighting matches the trade duration (day_trade emphasizes technicals and headlines; position_trade emphasizes fundamentals, industry, and macro). Overall Sentiment must not be flipped solely to agree with the requested trade direction.

    Set compliant to true only when both the template and faithfulness are acceptable.
    Set faithfulness to 0 when claims are largely unsupported, 1 when support is partial, and 2 when the synthesis stays within the specialist analyses.
    If the output is noncompliant, feedback must say what to revise. If it is compliant, feedback may be empty.
"""

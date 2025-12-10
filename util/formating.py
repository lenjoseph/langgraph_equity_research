from pydantic import BaseModel


def format_sentiment_output(output: BaseModel) -> str:
    """Format a sentiment output model as readable text."""
    lines = []
    data = output.model_dump()

    # Get the main sentiment/valuation field
    if "sentiment" in data:
        lines.append(f"[{data['sentiment']}]")
    elif "valuation" in data:
        lines.append(f"[{data['valuation']}]")

    lines.append("")

    # Format key points (standard agents)
    for kp in data.get("key_points", []):
        if isinstance(kp, dict):
            # KeyPointWithCitation
            lines.append(f"* {kp['point']} [{kp['source']}, {kp['date']}]")
        else:
            # Simple string key point
            lines.append(f"* {kp}")

    # Format key findings (filings agent)
    for finding in data.get("key_findings", []):
        lines.append(f"* {finding}")

    # Format citations (filings agent)
    citations = data.get("citations", [])
    if citations:
        lines.append("")
        lines.append("**Sources:**")
        for cite in citations:
            if isinstance(cite, dict):
                lines.append(
                    f"  - \"{cite.get('quote', '')}\" "
                    f"[{cite.get('filing_type', '')} {cite.get('section', '')}, {cite.get('filing_date', '')}]"
                )

    # Format risk factors summary (filings agent)
    risk_summary = data.get("risk_factors_summary")
    if risk_summary:
        lines.append("")
        lines.append("**Risk Factors:**")
        lines.append(risk_summary)

    lines.append("")
    lines.append(f"Confidence: {data.get('confidence', 'N/A')}")

    return "\n".join(lines)

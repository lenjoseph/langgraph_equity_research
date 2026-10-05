"""Offline checks for the shared synthesis contract and evaluator failure policy."""

from types import SimpleNamespace

from agents.aggregation.agent import build_aggregation_prompt
from agents.aggregation.prompt import research_aggregation_prompt
from agents.aggregation.rubric import REQUIRED_SECTIONS
from agents.evaluation.format_check import check_aggregation_format
from agents.evaluation.policy import evaluation_failure_update
from agents.shared.grounding import filter_ungrounded_points
from models.agent import Confidence, IndustrySentimentOutput, KeyPointWithCitation, RelativeSentiment
from models.result import GroundingSource
from models.state import EquityResearchState, TradeDirection, TradeDuration

GOOD_SYNTHESIS = """
**Summary of Research Findings:**
- Fundamental: margins are stable and cash flow covers debt
- Technical: momentum is flat
- Macro: growth is steady and policy is unchanged
- Industry: demand is healthy
- Peer: the company trades in line with peers
- Headline: recent news is quiet
- SEC Filings: risk factors are unchanged

Consensus and Divergence:
- Consensus: the business is stable
- Divergence: valuation views differ

Weighting of Perspectives:
- Fundamental 30 percent for a position trade
- Industry 20 percent
- Peer 10 percent
- Headline 5 percent
- Macro 20 percent
- Technical 5 percent
- SEC Filings 10 percent

**Overall Sentiment:** NEUTRAL

**Position Implication:** A long position is not strongly supported by this evidence.

**Conclusion:** The evidence is mixed, so the stance stays neutral and does not follow the requested side.
"""

BEARISH_SYNTHESIS = GOOD_SYNTHESIS.replace("NEUTRAL", "BEARISH").replace(
    "A long position is not strongly supported by this evidence.",
    "A short position has some support, but the label stays tied to the evidence.",
)


def test_good_synthesis_passes_format_contract():
    assert check_aggregation_format(GOOD_SYNTHESIS) == []


def test_missing_peer_section_fails_format_contract():
    broken = GOOD_SYNTHESIS.replace("- Peer: the company trades in line with peers\n", "")
    issues = check_aggregation_format(broken)
    assert any("Peer" in issue for issue in issues)


def test_required_sections_match_the_shared_template():
    assert "Peer:" in REQUIRED_SECTIONS
    assert "Position Implication:" in REQUIRED_SECTIONS
    assert "seven specialized research agents" in research_aggregation_prompt
    assert "day_rade" not in research_aggregation_prompt
    assert "day_trade" in research_aggregation_prompt


def test_fixture_set_matches_reference_labels():
    fixtures = [
        (GOOD_SYNTHESIS, "NEUTRAL", ["Fundamental", "SEC Filings", "Position Implication"]),
        (BEARISH_SYNTHESIS, "BEARISH", ["Fundamental", "SEC Filings", "Position Implication"]),
    ]
    for text, label, must_mention in fixtures:
        assert check_aggregation_format(text) == []
        assert f"Overall Sentiment:** {label}" in text or f"Overall Sentiment: {label}" in text
        for item in must_mention:
            assert item in text


def test_evaluator_failure_does_not_mark_draft_compliant():
    first = evaluation_failure_update(1, "Evaluation unavailable due to an internal error.")
    assert first["compliant"] is False
    assert first["evaluation_status"] == "failed"

    exhausted = evaluation_failure_update(3, "Evaluation skipped due to an internal error.")
    assert exhausted["compliant"] is False
    assert exhausted["evaluation_status"] == "skipped"


def test_revision_prompt_keeps_specialist_research():
    state = EquityResearchState(
        ticker="AAPL",
        trade_duration=TradeDuration.DAY_TRADE,
        trade_direction=TradeDirection.SHORT,
        fundamental_sentiment="cash flow is strong",
        feedback="Add the peer section.",
        combined_sentiment="prior draft",
    )
    prompt = build_aggregation_prompt(state)
    assert "cash flow is strong" in prompt
    assert "Fundamental Analysis:" in prompt
    assert "Add the peer section." in prompt
    assert "prior draft" in prompt
    assert "day_trade" in prompt


def test_ungrounded_citations_are_dropped():
    output = IndustrySentimentOutput(
        sentiment=RelativeSentiment.POSITIVE,
        key_points=[
            KeyPointWithCitation(point="Demand is up", source="Reuters", date="2026-01-01"),
            KeyPointWithCitation(point="Made up", source="Invented Daily", date="2026-01-01"),
        ],
        confidence=Confidence.MEDIUM,
    )
    filtered, dropped = filter_ungrounded_points(
        output, [GroundingSource(title="Reuters", uri="https://reuters.com/story")]
    )
    assert dropped == 1
    assert [point.source for point in filtered.key_points] == ["Reuters"]

    cleared, dropped_all = filter_ungrounded_points(output, [])
    assert dropped_all == 2
    assert cleared.key_points == []


def test_prompt_builder_accepts_mapping_like_state():
    state = SimpleNamespace(
        ticker="MSFT",
        trade_duration="swing_trade",
        trade_direction="long",
        fundamental_sentiment="a",
        technical_sentiment="b",
        macro_sentiment="c",
        peer_sentiment="d",
        industry_sentiment="e",
        headline_sentiment="f",
        filings_sentiment="g",
        feedback=None,
        combined_sentiment="",
    )
    prompt = build_aggregation_prompt(state)
    assert "swing_trade" in prompt
    assert "Fundamental Analysis:\na" in prompt

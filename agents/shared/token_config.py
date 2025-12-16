"""Token budget configuration for agents.

This module provides centralized configuration for token limits across all agents.
Token budgets can be set per-agent or globally for the entire request.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AgentTokenConfig(BaseModel):
    """Token configuration for a single agent."""

    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum output tokens per LLM call (None = no limit)",
    )
    token_budget: Optional[int] = Field(
        default=None,
        description="Maximum total tokens (input + output) for agent execution (None = no limit)",
    )


class TokenBudgetConfig(BaseModel):
    """Global token budget configuration for all agents."""

    # Global request-level budget
    request_budget: Optional[int] = Field(
        default=None,
        description="Maximum total tokens for entire request (None = no limit)",
    )

    # Per-agent configurations
    fundamental: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=5000,
        )
    )
    technical: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=2000,
        )
    )
    macro: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=2500,
        )
    )
    industry: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=2000,
        )
    )
    peer: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=2000,
        )
    )
    headline: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=500,
            token_budget=2000,
        )
    )
    filings_query_builder: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=2000,
            token_budget=4500,
        )
    )
    filings_synthesis: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=2048,
            token_budget=10000,
        )
    )
    aggregation: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=1000,
            token_budget=4000,
        )
    )
    evaluation: AgentTokenConfig = Field(
        default_factory=lambda: AgentTokenConfig(
            max_output_tokens=700,
            token_budget=4000,
        )
    )


# Default configuration instance - can be overridden per request
DEFAULT_TOKEN_CONFIG = TokenBudgetConfig()

# Preset configurations for different use cases
BUDGET_PRESETS = {
    # Unlimited - no token restrictions
    "unlimited": TokenBudgetConfig(
        request_budget=None,
        fundamental=AgentTokenConfig(),
        technical=AgentTokenConfig(),
        macro=AgentTokenConfig(),
        industry=AgentTokenConfig(),
        peer=AgentTokenConfig(),
        headline=AgentTokenConfig(),
        filings_query_builder=AgentTokenConfig(),
        filings_synthesis=AgentTokenConfig(),
        aggregation=AgentTokenConfig(),
        evaluation=AgentTokenConfig(),
    ),
    # Economy - minimal token usage for cost efficiency
    "economy": TokenBudgetConfig(
        request_budget=50000,
        fundamental=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        technical=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        macro=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        industry=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        peer=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        headline=AgentTokenConfig(max_output_tokens=1024, token_budget=4000),
        filings_query_builder=AgentTokenConfig(
            max_output_tokens=256, token_budget=1000
        ),
        filings_synthesis=AgentTokenConfig(max_output_tokens=1024, token_budget=5000),
        aggregation=AgentTokenConfig(max_output_tokens=2048, token_budget=10000),
        evaluation=AgentTokenConfig(max_output_tokens=512, token_budget=2500),
    ),
    # Standard - balanced token usage (default)
    "standard": DEFAULT_TOKEN_CONFIG,
    # Premium - higher limits for more detailed analysis
    "premium": TokenBudgetConfig(
        request_budget=200000,
        fundamental=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        technical=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        macro=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        industry=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        peer=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        headline=AgentTokenConfig(max_output_tokens=4096, token_budget=16000),
        filings_query_builder=AgentTokenConfig(
            max_output_tokens=1024, token_budget=4000
        ),
        filings_synthesis=AgentTokenConfig(max_output_tokens=4096, token_budget=20000),
        aggregation=AgentTokenConfig(max_output_tokens=8192, token_budget=40000),
        evaluation=AgentTokenConfig(max_output_tokens=2048, token_budget=10000),
    ),
}


def get_token_config(preset: str = "standard") -> TokenBudgetConfig:
    """Get token configuration by preset name.

    Args:
        preset: One of 'unlimited', 'economy', 'standard', 'premium'

    Returns:
        TokenBudgetConfig for the specified preset

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset not in BUDGET_PRESETS:
        raise ValueError(
            f"Unknown token preset: {preset}. "
            f"Available presets: {list(BUDGET_PRESETS.keys())}"
        )
    return BUDGET_PRESETS[preset]

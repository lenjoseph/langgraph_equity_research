from pydantic import BaseModel, Field


class AggregatorFeedback(BaseModel):
    compliant: bool = Field(
        description="Determine if the provided sentiment complies with the output structure"
    )
    feedback: str = Field(
        description="if the output is noncompliant, provide feedback on how to address the delta. Otherwise do not include feedback"
    )

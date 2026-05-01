from typing import Literal

from pydantic import BaseModel, Field


AllowedStatus = Literal["replied", "escalated"]
AllowedRequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
AllowedCompany = Literal["HackerRank", "Claude", "Visa", "None"]


class TicketInput(BaseModel):
    issue: str = Field(min_length=1)
    subject: str = Field(default="")
    company: AllowedCompany


class TicketOutput(BaseModel):
    status: AllowedStatus
    product_area: str = Field(min_length=1)
    response: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    request_type: AllowedRequestType


class RetrievedChunk(BaseModel):
    company: str
    product_area_hint: str
    source_path: str
    title: str
    section_heading: str
    chunk_text: str
    score: float = 0.0


class GuardrailDecision(BaseModel):
    must_escalate: bool
    reasons: list[str]


class DraftedOutput(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    citations: list[str] = []
    confidence: float = 0.0

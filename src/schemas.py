from __future__ import annotations
from pydantic import BaseModel, Field

class SalesFinding(BaseModel):
    title: str = Field(..., min_length=5)
    exec_summary: str = Field(..., min_length=60)
    kpis_used: list[str] = Field(..., min_length=1)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(..., min_length=2)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(..., min_length=1)

class CrewEnvelope(BaseModel):
    finding: SalesFinding

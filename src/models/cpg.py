"""
Clinical Practice Guideline (CPG) data models.
Enforces the mandatory template structure.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


class Recommendation(BaseModel):
    number: int
    statement: str
    evidence_level: EvidenceLevel
    strength: str = Field(description="e.g. Strong, Conditional")
    supporting_references: list[str] = Field(default_factory=list)


class RiskFactor(BaseModel):
    factor: str
    severity: str
    mitigation: str


class Contraindication(BaseModel):
    condition: str
    rationale: str
    alternatives: list[str] = Field(default_factory=list)


class MonitoringItem(BaseModel):
    parameter: str
    frequency: str
    target_range: Optional[str] = None
    action_if_abnormal: Optional[str] = None


class CPGDocument(BaseModel):
    """Complete Clinical Practice Guideline following the mandatory template."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Section 1
    title: str
    # Section 2
    executive_summary: str
    # Section 3
    scope_and_purpose: str
    # Section 4
    target_population: str
    # Section 5
    clinical_recommendations: list[Recommendation] = Field(default_factory=list)
    # Section 6
    evidence_summary: str
    # Section 7
    risk_assessment: list[RiskFactor] = Field(default_factory=list)
    # Section 8
    contraindications: list[Contraindication] = Field(default_factory=list)
    # Section 9
    monitoring_and_followup: list[MonitoringItem] = Field(default_factory=list)
    # Section 10
    references: list[str] = Field(default_factory=list)
    # Section 11
    review_date: Optional[date] = None
    authors: list[str] = Field(default_factory=list)
    specialty: Optional[str] = None


class CPGSectionUpdate(BaseModel):
    """Describes a targeted update to one or more CPG sections."""

    section_name: str = Field(description="The CPG section to modify")
    instruction: str = Field(description="What change to apply")
    new_content: Optional[str] = Field(
        None, description="Replacement content if fully rewritten"
    )


class CPGVersion(BaseModel):
    """Tracks a version snapshot of a CPG document."""

    version: int
    document: CPGDocument
    change_summary: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

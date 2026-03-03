"""
Shared test fixtures.
"""

import os
import pytest
import pytest_asyncio

# Set required env vars BEFORE any src imports so pydantic-settings won't fail
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "true")

from src.models.cpg import (
    CPGDocument,
    Contraindication,
    EvidenceLevel,
    MonitoringItem,
    Recommendation,
    RiskFactor,
)
from src.services.session_store import InMemorySessionStore


@pytest.fixture
def sample_cpg() -> CPGDocument:
    """A valid CPG document for use in tests."""
    return CPGDocument(
        title="Management of Type 2 Diabetes in Elderly Patients",
        executive_summary="This guideline addresses T2DM management in patients aged 65+.",
        scope_and_purpose="To provide evidence-based recommendations for T2DM in the elderly.",
        target_population="Adults aged 65 years and older with Type 2 Diabetes.",
        clinical_recommendations=[
            Recommendation(
                number=1,
                statement="Metformin is recommended as first-line therapy.",
                evidence_level=EvidenceLevel.HIGH,
                strength="Strong",
                supporting_references=["ADA Standards 2024"],
            ),
            Recommendation(
                number=2,
                statement="HbA1c target of <7.5% for most elderly patients.",
                evidence_level=EvidenceLevel.MODERATE,
                strength="Conditional",
                supporting_references=["Geriatric Diabetes Guidelines 2023"],
            ),
        ],
        evidence_summary="Based on 12 RCTs and 5 meta-analyses.",
        risk_assessment=[
            RiskFactor(
                factor="Hypoglycemia",
                severity="High",
                mitigation="Avoid sulfonylureas; prefer DPP-4 inhibitors.",
            ),
        ],
        contraindications=[
            Contraindication(
                condition="Severe renal impairment (eGFR <30)",
                rationale="Metformin accumulation risk.",
                alternatives=["DPP-4 inhibitors", "Insulin"],
            ),
        ],
        monitoring_and_followup=[
            MonitoringItem(
                parameter="HbA1c",
                frequency="Every 3-6 months",
                target_range="<7.5%",
                action_if_abnormal="Intensify therapy or adjust targets.",
            ),
        ],
        references=["ADA Standards of Medical Care 2024"],
        specialty="Endocrinology",
        authors=["Test Author"],
    )


@pytest.fixture
def session_store() -> InMemorySessionStore:
    """A fresh in-memory session store."""
    return InMemorySessionStore()

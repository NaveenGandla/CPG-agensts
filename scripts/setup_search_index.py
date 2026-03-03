"""
Standalone script to create the Azure AI Search index and load sample documents.
Run: python -m scripts.setup_search_index
"""

import asyncio
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.models.search import SearchDocument
from src.services.search_client import ensure_index_exists, index_documents

SAMPLE_DOCUMENTS = [
    SearchDocument(
        id="doc-001",
        title="Management of Type 2 Diabetes in Adults",
        content=(
            "Type 2 diabetes management involves lifestyle modifications, "
            "metformin as first-line therapy, and individualized glycemic targets. "
            "HbA1c should be monitored every 3-6 months. SGLT2 inhibitors and "
            "GLP-1 receptor agonists are recommended for patients with cardiovascular "
            "or renal comorbidities."
        ),
        specialty="Endocrinology",
        publication_year=2024,
        evidence_level="High",
        source="ADA Standards of Medical Care 2024",
        abstract="Comprehensive guidelines for T2DM management in adults.",
        keywords=["diabetes", "metformin", "HbA1c", "SGLT2"],
    ),
    SearchDocument(
        id="doc-002",
        title="Hypertension Management Guidelines",
        content=(
            "Blood pressure targets of <130/80 mmHg are recommended for most adults. "
            "First-line agents include ACE inhibitors, ARBs, calcium channel blockers, "
            "and thiazide diuretics. Lifestyle modifications should accompany pharmacotherapy. "
            "Home blood pressure monitoring is encouraged."
        ),
        specialty="Cardiology",
        publication_year=2023,
        evidence_level="High",
        source="ACC/AHA Hypertension Guideline 2023",
        abstract="Evidence-based hypertension management recommendations.",
        keywords=["hypertension", "blood pressure", "ACE inhibitor", "ARB"],
    ),
    SearchDocument(
        id="doc-003",
        title="Anticoagulation in Atrial Fibrillation",
        content=(
            "DOACs are preferred over warfarin for stroke prevention in non-valvular AF. "
            "CHA2DS2-VASc score guides anticoagulation decisions. Bleeding risk should be "
            "assessed using HAS-BLED score. Renal function must be monitored regularly."
        ),
        specialty="Cardiology",
        publication_year=2024,
        evidence_level="High",
        source="ESC AF Guidelines 2024",
        abstract="Stroke prevention strategies in atrial fibrillation.",
        keywords=["atrial fibrillation", "anticoagulation", "DOAC", "stroke"],
    ),
]


async def main():
    print("Creating search index...")
    await ensure_index_exists()

    print(f"Indexing {len(SAMPLE_DOCUMENTS)} sample documents...")
    count = await index_documents(SAMPLE_DOCUMENTS)
    print(f"Successfully indexed {count} documents.")


if __name__ == "__main__":
    asyncio.run(main())

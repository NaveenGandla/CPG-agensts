"""
Unit tests for agent tools.

Azure services (AI Search, OpenAI) are mocked so tests run without
any cloud resources.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.tools import (
    retrieve_cpg_document,
    search_clinical_evidence,
    store_cpg_document,
    validate_cpg_structure,
)
from src.models.cpg import CPGDocument
from src.models.search import SearchResult
from src.services.session_store import InMemorySessionStore


# ── validate_cpg_structure ──────────────────────────────────────


class TestValidateCPGStructure:
    def test_valid_document_passes(self, sample_cpg):
        cpg_json = sample_cpg.model_dump_json()
        result = validate_cpg_structure(cpg_json)
        assert "compliant" in result.lower()

    def test_missing_sections_detected(self):
        incomplete = json.dumps({"title": "Test", "executive_summary": "Summary"})
        result = validate_cpg_structure(incomplete)
        assert "CRITICAL" in result
        assert "scope_and_purpose" in result

    def test_empty_sections_detected(self):
        doc = {
            "title": "Test",
            "executive_summary": "",
            "scope_and_purpose": "ok",
            "target_population": "ok",
            "clinical_recommendations": [],
            "evidence_summary": "ok",
            "risk_assessment": [{"factor": "f", "severity": "s", "mitigation": "m"}],
            "contraindications": [{"condition": "c", "rationale": "r"}],
            "monitoring_and_followup": [{"parameter": "p", "frequency": "f"}],
            "references": ["ref"],
        }
        result = validate_cpg_structure(json.dumps(doc))
        assert "MAJOR" in result
        assert "executive_summary" in result
        assert "clinical_recommendations" in result

    def test_invalid_json_handled(self):
        result = validate_cpg_structure("not json {{{")
        assert "Invalid JSON" in result


# ── search_clinical_evidence ────────────────────────────────────


class TestSearchClinicalEvidence:
    @pytest.mark.asyncio
    @patch("src.agents.tools.hybrid_search")
    async def test_returns_formatted_results(self, mock_search):
        mock_search.return_value = [
            SearchResult(
                id="1",
                title="Diabetes Guidelines",
                content="Metformin is first-line therapy.",
                score=0.95,
                specialty="Endocrinology",
                publication_year=2024,
                evidence_level="High",
                source="ADA 2024",
            ),
        ]

        result = await search_clinical_evidence.__wrapped__(
            query="diabetes management",
        )

        assert "Diabetes Guidelines" in result
        assert "Metformin" in result
        assert "ADA 2024" in result
        mock_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.agents.tools.hybrid_search")
    async def test_no_results_message(self, mock_search):
        mock_search.return_value = []

        result = await search_clinical_evidence.__wrapped__(
            query="obscure topic",
        )

        assert "No relevant clinical evidence" in result

    @pytest.mark.asyncio
    @patch("src.agents.tools.hybrid_search")
    async def test_filters_passed_through(self, mock_search):
        mock_search.return_value = []

        await search_clinical_evidence.__wrapped__(
            query="hypertension",
            specialty="Cardiology",
            min_publication_year=2020,
            evidence_level="High",
        )

        call_args = mock_search.call_args[0][0]
        assert call_args.filter.specialty == "Cardiology"
        assert call_args.filter.min_publication_year == 2020
        assert call_args.filter.evidence_level == "High"


# ── store_cpg_document ──────────────────────────────────────────


class TestStoreCPGDocument:
    @pytest.mark.asyncio
    async def test_stores_valid_cpg(self, sample_cpg):
        store = InMemorySessionStore()
        cpg_json = sample_cpg.model_dump_json()

        result = await store_cpg_document.__wrapped__(
            cpg_json=cpg_json,
            session_store=store,
        )

        assert "stored successfully" in result
        assert sample_cpg.id in result
        retrieved = await store.get_cpg(sample_cpg.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_error_without_store(self, sample_cpg):
        result = await store_cpg_document.__wrapped__(
            cpg_json=sample_cpg.model_dump_json(),
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_error_on_invalid_json(self):
        store = InMemorySessionStore()
        result = await store_cpg_document.__wrapped__(
            cpg_json="not valid json",
            session_store=store,
        )
        assert "Error" in result


# ── retrieve_cpg_document ───────────────────────────────────────


class TestRetrieveCPGDocument:
    @pytest.mark.asyncio
    async def test_retrieves_existing_cpg(self, sample_cpg):
        store = InMemorySessionStore()
        await store.save_cpg(sample_cpg)

        result = await retrieve_cpg_document.__wrapped__(
            cpg_id=sample_cpg.id,
            session_store=store,
        )

        data = json.loads(result)
        assert data["title"] == sample_cpg.title

    @pytest.mark.asyncio
    async def test_not_found_message(self):
        store = InMemorySessionStore()
        result = await retrieve_cpg_document.__wrapped__(
            cpg_id="nonexistent",
            session_store=store,
        )
        assert "No CPG document found" in result

    @pytest.mark.asyncio
    async def test_error_without_store(self):
        result = await retrieve_cpg_document.__wrapped__(
            cpg_id="any",
        )
        assert "Error" in result

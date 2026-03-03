"""
Unit tests for Pydantic data models.
"""

import json
from datetime import date

import pytest

from src.models.cpg import (
    CPGDocument,
    CPGSectionUpdate,
    CPGVersion,
    Contraindication,
    EvidenceLevel,
    MonitoringItem,
    Recommendation,
    RiskFactor,
)
from src.models.search import SearchDocument, SearchFilter, SearchQuery, SearchResult
from src.models.session import SessionMetadata


# ── CPG Models ──────────────────────────────────────────────────


class TestCPGDocument:
    def test_creates_with_defaults(self):
        cpg = CPGDocument(
            title="Test",
            executive_summary="Summary",
            scope_and_purpose="Purpose",
            target_population="Adults",
            evidence_summary="Evidence",
        )
        assert cpg.id  # auto-generated UUID
        assert cpg.version == 1
        assert cpg.title == "Test"
        assert cpg.clinical_recommendations == []
        assert cpg.references == []

    def test_full_document_roundtrip(self, sample_cpg):
        """Serialize to JSON and back."""
        json_str = sample_cpg.model_dump_json()
        restored = CPGDocument.model_validate_json(json_str)
        assert restored.title == sample_cpg.title
        assert len(restored.clinical_recommendations) == 2
        assert restored.clinical_recommendations[0].evidence_level == EvidenceLevel.HIGH
        assert len(restored.contraindications) == 1
        assert restored.specialty == "Endocrinology"

    def test_version_increment(self, sample_cpg):
        sample_cpg.version = 2
        assert sample_cpg.version == 2

    def test_json_mode_serialization(self, sample_cpg):
        """Ensure mode='json' produces JSON-safe types (no datetime objects)."""
        data = sample_cpg.model_dump(mode="json")
        # Should be JSON-serializable without errors
        json_str = json.dumps(data)
        assert '"title"' in json_str


class TestRecommendation:
    def test_evidence_levels(self):
        for level in EvidenceLevel:
            rec = Recommendation(
                number=1,
                statement="Test",
                evidence_level=level,
                strength="Strong",
            )
            assert rec.evidence_level == level

    def test_invalid_evidence_level_rejected(self):
        with pytest.raises(ValueError):
            Recommendation(
                number=1,
                statement="Test",
                evidence_level="Invalid",
                strength="Strong",
            )


class TestCPGVersion:
    def test_version_snapshot(self, sample_cpg):
        version = CPGVersion(
            version=1,
            document=sample_cpg,
            change_summary="Initial version",
        )
        assert version.version == 1
        assert version.document.title == sample_cpg.title


# ── Search Models ───────────────────────────────────────────────


class TestSearchFilter:
    def test_all_none_by_default(self):
        f = SearchFilter()
        assert f.specialty is None
        assert f.min_publication_year is None

    def test_with_values(self):
        f = SearchFilter(specialty="Cardiology", min_publication_year=2020)
        assert f.specialty == "Cardiology"
        assert f.min_publication_year == 2020


class TestSearchQuery:
    def test_defaults(self):
        q = SearchQuery(query_text="diabetes")
        assert q.top_k == 10
        assert q.use_semantic_ranking is True
        assert q.use_vector_search is True


class TestSearchDocument:
    def test_minimal_document(self):
        doc = SearchDocument(
            id="doc-1",
            title="Test",
            content="Content",
            specialty="General",
            publication_year=2024,
            evidence_level="High",
            source="Test Source",
        )
        assert doc.content_vector == []
        assert doc.keywords == []


# ── Session Models ──────────────────────────────────────────────


class TestSessionMetadata:
    def test_defaults(self):
        meta = SessionMetadata()
        assert meta.id  # auto-generated
        assert meta.current_cpg_id is None
        assert meta.cpg_version_ids == []

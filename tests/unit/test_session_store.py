"""
Unit tests for the in-memory session store.
"""

import pytest

from src.models.cpg import CPGDocument
from src.services.session_store import InMemorySessionStore


@pytest.fixture
def store():
    return InMemorySessionStore()


class TestInMemorySessionStore:
    @pytest.mark.asyncio
    async def test_save_and_get_cpg(self, store, sample_cpg):
        await store.save_cpg(sample_cpg)
        retrieved = await store.get_cpg(sample_cpg.id)
        assert retrieved is not None
        assert retrieved.title == sample_cpg.title
        assert retrieved.id == sample_cpg.id

    @pytest.mark.asyncio
    async def test_get_missing_cpg_returns_none(self, store):
        result = await store.get_cpg("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_version_tracking(self, store, sample_cpg):
        # Save version 1
        await store.save_cpg(sample_cpg)

        # Save version 2
        sample_cpg.version = 2
        sample_cpg.target_population = "Updated population"
        await store.save_cpg(sample_cpg)

        versions = await store.get_cpg_versions(sample_cpg.id)
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    @pytest.mark.asyncio
    async def test_get_versions_empty_for_unknown_id(self, store):
        versions = await store.get_cpg_versions("unknown")
        assert versions == []

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, store, sample_cpg):
        await store.save_cpg(sample_cpg)
        sample_cpg.title = "Updated Title"
        await store.save_cpg(sample_cpg)

        retrieved = await store.get_cpg(sample_cpg.id)
        assert retrieved.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_session_crud(self, store):
        from src.models.session import SessionMetadata

        # No built-in session CRUD in current store, but test CPG isolation
        cpg1 = CPGDocument(
            title="CPG 1",
            executive_summary="s",
            scope_and_purpose="s",
            target_population="s",
            evidence_summary="s",
        )
        cpg2 = CPGDocument(
            title="CPG 2",
            executive_summary="s",
            scope_and_purpose="s",
            target_population="s",
            evidence_summary="s",
        )
        await store.save_cpg(cpg1)
        await store.save_cpg(cpg2)

        assert (await store.get_cpg(cpg1.id)).title == "CPG 1"
        assert (await store.get_cpg(cpg2.id)).title == "CPG 2"

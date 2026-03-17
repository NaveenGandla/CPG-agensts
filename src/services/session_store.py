"""
Session state management.
Uses Azure Cosmos DB when available, falls back to in-memory store.
"""

import json
from typing import Optional

from src.core.logging import get_logger
from src.core.settings import get_settings
from src.models.cpg import CPGDocument, CPGVersion
from src.models.session import SessionMetadata as Session

logger = get_logger("session_store")


class InMemorySessionStore:
    """Thread-safe in-memory session store (development / single-instance)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._cpg_docs: dict[str, CPGDocument] = {}
        self._cpg_versions: dict[str, list[CPGVersion]] = {}

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def save_session(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def get_cpg(self, cpg_id: str) -> Optional[CPGDocument]:
        return self._cpg_docs.get(cpg_id)

    async def save_cpg(self, cpg: CPGDocument) -> None:
        self._cpg_docs[cpg.id] = cpg
        # Store version snapshot
        versions = self._cpg_versions.setdefault(cpg.id, [])
        versions.append(
            CPGVersion(
                version=cpg.version,
                document=cpg.model_copy(deep=True),
                change_summary=f"Version {cpg.version}",
            )
        )

    async def get_cpg_versions(self, cpg_id: str) -> list[CPGVersion]:
        return self._cpg_versions.get(cpg_id, [])


class CosmosSessionStore:
    """Azure Cosmos DB session store for production multi-instance deployments."""

    def __init__(self) -> None:
        self._container = None

    async def _get_container(self):
        if self._container is None:
            from azure.cosmos.aio import CosmosClient

            settings = get_settings()
            cosmos = settings.azure_cosmos

            if settings.use_key_auth:
                if not cosmos.key:
                    raise ValueError(
                        "AZURE_COSMOS_KEY is required when AUTH_MODE=key"
                    )
                client = CosmosClient(cosmos.endpoint, credential=cosmos.key)
            else:
                from azure.identity.aio import DefaultAzureCredential

                credential = DefaultAzureCredential()
                client = CosmosClient(cosmos.endpoint, credential=credential)

            database = client.get_database_client(cosmos.database_name)
            self._container = database.get_container_client(cosmos.container_name)
        return self._container

    async def get_session(self, session_id: str) -> Optional[Session]:
        container = await self._get_container()
        try:
            item = await container.read_item(
                item=f"session:{session_id}",
                partition_key=f"session:{session_id}",
            )
            return Session.model_validate_json(item["data"])
        except Exception:
            return None

    async def save_session(self, session: Session) -> None:
        container = await self._get_container()
        ttl = get_settings().azure_cosmos.session_ttl_seconds
        await container.upsert_item({
            "id": f"session:{session.id}",
            "type": "session",
            "data": session.model_dump_json(),
            "ttl": ttl,
        })

    async def delete_session(self, session_id: str) -> None:
        container = await self._get_container()
        try:
            await container.delete_item(
                item=f"session:{session_id}",
                partition_key=f"session:{session_id}",
            )
        except Exception:
            pass

    async def get_cpg(self, cpg_id: str) -> Optional[CPGDocument]:
        container = await self._get_container()
        try:
            item = await container.read_item(
                item=f"cpg:{cpg_id}",
                partition_key=f"cpg:{cpg_id}",
            )
            return CPGDocument.model_validate_json(item["data"])
        except Exception:
            return None

    async def save_cpg(self, cpg: CPGDocument) -> None:
        container = await self._get_container()
        # Save current CPG document
        await container.upsert_item({
            "id": f"cpg:{cpg.id}",
            "type": "cpg",
            "data": cpg.model_dump_json(),
        })
        # Save version snapshot
        version = CPGVersion(
            version=cpg.version,
            document=cpg.model_copy(deep=True),
            change_summary=f"Version {cpg.version}",
        )
        await container.upsert_item({
            "id": f"cpg_version:{cpg.id}:v{cpg.version}",
            "type": "cpg_version",
            "cpg_id": cpg.id,
            "data": version.model_dump_json(),
        })

    async def get_cpg_versions(self, cpg_id: str) -> list[CPGVersion]:
        container = await self._get_container()
        query = (
            "SELECT * FROM c WHERE c.type = 'cpg_version' "
            "AND c.cpg_id = @cpg_id ORDER BY c.id ASC"
        )
        items = container.query_items(
            query=query,
            parameters=[{"name": "@cpg_id", "value": cpg_id}],
        )
        versions = []
        async for item in items:
            versions.append(CPGVersion.model_validate_json(item["data"]))
        return versions


def create_session_store() -> InMemorySessionStore | CosmosSessionStore:
    """Factory: use Cosmos DB when configured, otherwise in-memory."""
    settings = get_settings().azure_cosmos
    if settings.endpoint:
        logger.info("Using Cosmos DB session store at %s", settings.endpoint)
        return CosmosSessionStore()
    logger.info("Using in-memory session store (no Cosmos DB configured).")
    return InMemorySessionStore()

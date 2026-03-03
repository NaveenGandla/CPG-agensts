"""
Session state management.
Uses Azure Redis Cache when available, falls back to in-memory store.
"""

import json
from typing import Optional

from src.core.logging import get_logger
from src.core.settings import get_settings
from src.models.cpg import CPGDocument, CPGVersion
from src.models.session import Session

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


class RedisSessionStore:
    """Azure Redis Cache session store for production multi-instance deployments."""

    def __init__(self) -> None:
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as redis

            settings = get_settings().azure_redis
            self._redis = redis.Redis(
                host=settings.host,
                port=settings.port,
                ssl=settings.ssl,
                decode_responses=True,
            )
        return self._redis

    async def get_session(self, session_id: str) -> Optional[Session]:
        r = await self._get_redis()
        data = await r.get(f"session:{session_id}")
        if data:
            return Session.model_validate_json(data)
        return None

    async def save_session(self, session: Session) -> None:
        r = await self._get_redis()
        ttl = get_settings().azure_redis.session_ttl_seconds
        await r.setex(
            f"session:{session.id}", ttl, session.model_dump_json()
        )

    async def delete_session(self, session_id: str) -> None:
        r = await self._get_redis()
        await r.delete(f"session:{session_id}")

    async def get_cpg(self, cpg_id: str) -> Optional[CPGDocument]:
        r = await self._get_redis()
        data = await r.get(f"cpg:{cpg_id}")
        if data:
            return CPGDocument.model_validate_json(data)
        return None

    async def save_cpg(self, cpg: CPGDocument) -> None:
        r = await self._get_redis()
        await r.set(f"cpg:{cpg.id}", cpg.model_dump_json())
        # Push version
        version = CPGVersion(
            version=cpg.version,
            document=cpg.model_copy(deep=True),
            change_summary=f"Version {cpg.version}",
        )
        await r.rpush(f"cpg_versions:{cpg.id}", version.model_dump_json())

    async def get_cpg_versions(self, cpg_id: str) -> list[CPGVersion]:
        r = await self._get_redis()
        raw = await r.lrange(f"cpg_versions:{cpg_id}", 0, -1)
        return [CPGVersion.model_validate_json(v) for v in raw]


def create_session_store() -> InMemorySessionStore | RedisSessionStore:
    """Factory: use Redis when configured, otherwise in-memory."""
    settings = get_settings().azure_redis
    if settings.host:
        logger.info("Using Redis session store at %s", settings.host)
        return RedisSessionStore()
    logger.info("Using in-memory session store (no Redis configured).")
    return InMemorySessionStore()

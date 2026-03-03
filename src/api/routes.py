"""
FastAPI route definitions.

Uses Microsoft Agent Framework agents with AgentSession for
conversation state and tool-based orchestration.
"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from src.agents.factory import create_orchestrator_agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CPGResponse,
    CPGVersionListResponse,
    HealthResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
)
from src.core.logging import get_logger
from src.core.settings import get_settings
from src.models.search import SearchDocument
from src.services.search_client import index_documents
from src.services.session_store import create_session_store

logger = get_logger("api")

router = APIRouter()

# ── Shared state — initialised lazily on first request ──────────

_session_store = None
_orchestrator = None
_sessions: dict = {}  # session_id → AgentSession


def _ensure_initialised():
    """Lazily create the session store, then the orchestrator agent with workflow."""
    global _session_store, _orchestrator
    if _orchestrator is None:
        _session_store = create_session_store()
        # Pass session_store so the workflow pipeline tool can persist CPGs
        _orchestrator = create_orchestrator_agent(session_store=_session_store)
        logger.info("Orchestrator agent and session store initialised.")


# ──────────────────────────── Health ────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
    )


# ──────────────────────────── Chat ──────────────────────────────


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the CPG multi-agent system.

    The orchestrator agent uses AgentSession for conversation continuity.
    It has tools for evidence search, CPG generation (via sub-agent),
    CPG review (via sub-agent), validation, and document storage.
    """
    _ensure_initialised()
    session_id = request.session_id or str(uuid4())

    # Retrieve or create AgentSession for this conversation
    if session_id in _sessions:
        session = _sessions[session_id]
    else:
        session = _orchestrator.create_session()
        _sessions[session_id] = session

    try:
        # Run the orchestrator agent with the user message.
        # The agent decides which tools to call (search, generate, modify, etc.)
        # The session_store kwarg is injected into tools that accept **kwargs.
        result = await _orchestrator.run(
            request.message,
            session=session,
            session_store=_session_store,
        )

        response_text = result.text if result.text else ""

        return ChatResponse(
            session_id=session_id,
            response=response_text,
        )

    except Exception as exc:
        logger.exception("Agent execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────── CPG ───────────────────────────────


@router.get("/cpg/{cpg_id}", response_model=CPGResponse, tags=["cpg"])
async def get_cpg(cpg_id: str):
    """Retrieve a CPG document by ID."""
    _ensure_initialised()

    cpg = await _session_store.get_cpg(cpg_id)
    if cpg is None:
        raise HTTPException(status_code=404, detail="CPG not found")

    return CPGResponse(
        cpg_id=cpg.id,
        version=cpg.version,
        cpg=cpg.model_dump(mode="json"),
    )


@router.get(
    "/cpg/{cpg_id}/versions",
    response_model=CPGVersionListResponse,
    tags=["cpg"],
)
async def get_cpg_versions(cpg_id: str):
    """List all versions of a CPG document."""
    _ensure_initialised()

    versions = await _session_store.get_cpg_versions(cpg_id)
    return CPGVersionListResponse(
        cpg_id=cpg_id,
        versions=[v.model_dump(mode="json") for v in versions],
    )


# ──────────────────────────── Indexing ──────────────────────────


@router.post(
    "/index", response_model=IndexDocumentResponse, tags=["indexing"]
)
async def index_docs(request: IndexDocumentRequest):
    """Index clinical documents into Azure AI Search."""
    docs = [SearchDocument(**d) for d in request.documents]
    count = await index_documents(docs)
    return IndexDocumentResponse(
        indexed_count=count,
        total_submitted=len(docs),
    )

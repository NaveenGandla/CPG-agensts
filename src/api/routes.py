"""
FastAPI route definitions.

Uses Microsoft Agent Framework agents with AgentSession for
conversation state and tool-based orchestration.
"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile

from src.agents.factory import create_orchestrator_agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CPGResponse,
    CPGVersionListResponse,
    DocumentListItem,
    DocumentListResponse,
    HealthResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TemplateUploadResponse,
)
from src.core.logging import get_logger
from src.core.settings import get_settings
from src.models.search import SearchDocument
from src.models.search import SearchFilter, SearchQuery
from src.services.search_client import hybrid_search, index_documents
from src.services.session_store import create_session_store
from src.services.template_parser import parse_template

logger = get_logger("api")

router = APIRouter()

# ── Shared state — initialised lazily on first request ──────────

_session_store = None
_orchestrator = None
_sessions: dict = {}  # session_id → AgentSession
_session_templates: dict = {}  # session_id → template instructions string


def _ensure_initialised():
    """Lazily create the session store and the orchestrator agent."""
    global _session_store, _orchestrator
    if _orchestrator is None:
        _session_store = create_session_store()
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


# ──────────────────────────── Template Upload ─────────────────


@router.post(
    "/template/upload",
    response_model=TemplateUploadResponse,
    tags=["template"],
)
async def upload_template(
    file: UploadFile,
    session_id: str | None = None,
):
    """
    Upload a CPG template file (PDF, DOCX, PPTX, XLSX, TXT).

    The file is parsed using Azure Document Intelligence to extract its
    structure (sections, headings, tables). The extracted template is stored
    for the session and automatically injected into all subsequent chat
    messages so the agent generates CPGs in the same format.
    """
    _ensure_initialised()
    session_id = session_id or str(uuid4())

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(file_bytes) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    try:
        parsed = await parse_template(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Template parsing failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Template parsing failed: {exc}")

    # Store the parsed template instructions for this session
    _session_templates[session_id] = parsed["template_instructions"]
    logger.info(
        "Template uploaded for session %s: %s (%d sections)",
        session_id, file.filename, len(parsed["sections"]),
    )

    # Ensure an AgentSession exists so the chat endpoint can use it
    if session_id not in _sessions:
        _sessions[session_id] = _orchestrator.create_session()

    return TemplateUploadResponse(
        session_id=session_id,
        filename=file.filename,
        sections=parsed["sections"],
        tables=parsed["tables"],
        message=(
            f"Template '{file.filename}' parsed successfully. "
            f"Found {len(parsed['sections'])} sections and "
            f"{len(parsed['tables'])} tables. "
            f"The CPG will be generated following this template's format."
        ),
    )


# ──────────────────────────── Chat ──────────────────────────────


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the CPG multi-agent system.

    If a template has been uploaded for this session, the template
    instructions are automatically prepended to the first message.
    """
    _ensure_initialised()
    session_id = request.session_id or str(uuid4())

    # Retrieve or create AgentSession for this conversation
    if session_id in _sessions:
        session = _sessions[session_id]
    else:
        session = _orchestrator.create_session()
        _sessions[session_id] = session

    # Inject template context if one was uploaded for this session
    message = request.message
    template_instructions = _session_templates.get(session_id)
    if template_instructions:
        message = (
            f"{message}\n\n"
            f"--- TEMPLATE CONTEXT ---\n"
            f"{template_instructions}\n"
            f"--- END TEMPLATE CONTEXT ---"
        )
        # Only inject on first use, then clear so it's not repeated
        del _session_templates[session_id]

    try:
        result = await _orchestrator.run(
            message,
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


# ──────────────────────────── Search ───────────────────────────


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search_evidence(request: SearchRequest):
    """Search the clinical knowledge base with filters."""
    search_filter = SearchFilter(
        specialty=request.specialty,
        min_publication_year=request.min_publication_year,
        evidence_level=request.evidence_level,
    )
    query = SearchQuery(
        query_text=request.query,
        filter=search_filter if any([
            request.specialty, request.min_publication_year, request.evidence_level,
        ]) else None,
        top_k=request.top_k,
        use_semantic_ranking=True,
        use_vector_search=True,
    )

    results = await hybrid_search(query)
    items = [
        SearchResultItem(
            id=r.id, title=r.title, content=r.content, score=r.score,
            specialty=r.specialty, publication_year=r.publication_year,
            evidence_level=r.evidence_level, source=r.source,
        )
        for r in results
    ]
    return SearchResponse(results=items, total=len(items))


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(
    specialty: str | None = None,
    top: int = 50,
):
    """List indexed documents from the knowledge base."""
    query = SearchQuery(
        query_text="*",
        filter=SearchFilter(specialty=specialty) if specialty else None,
        top_k=top,
        use_semantic_ranking=False,
        use_vector_search=False,
    )

    results = await hybrid_search(query)

    seen_titles: set[str] = set()
    docs: list[DocumentListItem] = []
    for r in results:
        if r.title in seen_titles:
            continue
        seen_titles.add(r.title)
        docs.append(DocumentListItem(
            id=r.id, title=r.title, specialty=r.specialty,
            publication_year=r.publication_year,
            evidence_level=r.evidence_level, source=r.source,
            content_preview=r.content[:200] if r.content else "",
        ))
    return DocumentListResponse(documents=docs, total=len(docs))


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

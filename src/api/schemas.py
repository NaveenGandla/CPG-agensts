"""
API request / response schemas.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Omit to start new session.",
    )
    message: str = Field(..., min_length=1, description="User message text.")


class ChatResponse(BaseModel):
    session_id: str
    response: str


class CPGResponse(BaseModel):
    cpg_id: str
    version: int
    cpg: dict[str, Any]


class CPGVersionListResponse(BaseModel):
    cpg_id: str
    versions: list[dict[str, Any]]


class IndexDocumentRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(
        ..., min_length=1, description="Documents to index."
    )


class IndexDocumentResponse(BaseModel):
    indexed_count: int
    total_submitted: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text.")
    specialty: Optional[str] = None
    min_publication_year: Optional[int] = None
    evidence_level: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)


class SearchResultItem(BaseModel):
    id: str
    title: str
    content: str
    score: float
    specialty: str
    publication_year: int
    evidence_level: str
    source: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


class DocumentListItem(BaseModel):
    id: str
    title: str
    specialty: str
    publication_year: int
    evidence_level: str
    source: str
    content_preview: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


class TemplateUploadResponse(BaseModel):
    session_id: str
    filename: str
    sections: list[dict[str, str]]
    tables: list[dict[str, Any]]
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    environment: str

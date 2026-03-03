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


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    environment: str

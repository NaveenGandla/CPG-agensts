"""
Azure AI Search document and query models.
"""

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchDocument(BaseModel):
    """Schema for documents stored in Azure AI Search."""

    id: str
    title: str
    content: str
    specialty: str
    publication_year: int
    evidence_level: str
    source: str
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    content_vector: list[float] = Field(
        default_factory=list, description="Embedding vector"
    )


class SearchFilter(BaseModel):
    """Metadata filters for AI Search queries."""

    specialty: Optional[str] = None
    min_publication_year: Optional[int] = None
    max_publication_year: Optional[int] = None
    evidence_level: Optional[str] = None


class SearchQuery(BaseModel):
    """Encapsulates a search request."""

    query_text: str
    filter: Optional[SearchFilter] = None
    top_k: int = 10
    use_semantic_ranking: bool = True
    use_vector_search: bool = True


class SearchResult(BaseModel):
    """A single search result with score and metadata."""

    id: str
    title: str
    content: str
    score: float
    specialty: str
    publication_year: int
    evidence_level: str
    source: str
    highlights: list[str] = Field(default_factory=list)

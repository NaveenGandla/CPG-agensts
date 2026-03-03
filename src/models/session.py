"""
Conversation session and message models.

NOTE: With Microsoft Agent Framework, conversation history is managed by
AgentSession. This module is retained for CPG document tracking and
any custom metadata that supplements the framework's session state.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """Tracks CPG-related state alongside the framework's AgentSession."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    current_cpg_id: Optional[str] = None
    cpg_version_ids: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

"""
Agent role definitions.

With Microsoft Agent Framework the inter-agent communication is handled
by the framework itself (AgentSession, Workflow edges, agent-as-tool).
These enums are retained for logging and observability.
"""

from enum import Enum


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    CPG_AUTHOR = "cpg_author"
    CPG_REVIEWER = "cpg_reviewer"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"

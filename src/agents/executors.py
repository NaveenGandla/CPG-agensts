"""
Workflow Executors for the CPG generation pipeline.

Each executor is a processing unit in the WorkflowBuilder graph.
The pipeline: KnowledgeRetrieval → CPGAuthoring → CPGReview
"""

import json
from typing import Any

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentResponseUpdate,
    Executor,
    WorkflowContext,
    handler,
)

from src.core.logging import get_logger
from src.models.search import SearchFilter, SearchQuery
from src.services.search_client import hybrid_search

logger = get_logger("executors")


# ── Message types flowing through the workflow ──────────────────


class RetrievalRequest:
    """Input to the knowledge retrieval step."""

    def __init__(self, topic: str, specialty: str | None = None, requirements: str = ""):
        self.topic = topic
        self.specialty = specialty
        self.requirements = requirements


class RetrievalResult:
    """Output from knowledge retrieval, input to CPG authoring."""

    def __init__(self, topic: str, evidence: str, requirements: str = ""):
        self.topic = topic
        self.evidence = evidence
        self.requirements = requirements


class CPGDraft:
    """Output from CPG authoring, input to review."""

    def __init__(self, cpg_json: str, topic: str):
        self.cpg_json = cpg_json
        self.topic = topic


class CPGReviewResult:
    """Final output from the review step."""

    def __init__(self, cpg_json: str, review_json: str):
        self.cpg_json = cpg_json
        self.review_json = review_json


# ── Knowledge Retrieval Executor ────────────────────────────────


class KnowledgeRetrievalExecutor(Executor):
    """Searches Azure AI Search for clinical evidence relevant to the topic."""

    @handler
    async def handle_retrieval(
        self, request: RetrievalRequest, ctx: WorkflowContext[RetrievalResult]
    ) -> None:
        logger.info("Retrieving evidence for topic: %s", request.topic[:100])

        search_filter = (
            SearchFilter(specialty=request.specialty)
            if request.specialty
            else None
        )
        query = SearchQuery(
            query_text=request.topic,
            filter=search_filter,
            top_k=10,
            use_semantic_ranking=True,
            use_vector_search=True,
        )

        results = await hybrid_search(query)

        evidence_parts = [
            f"[{r.title}] ({r.source}, {r.evidence_level}): {r.content[:500]}"
            for r in results[:5]
        ]
        evidence = "\n\n".join(evidence_parts) if evidence_parts else "No evidence found."

        logger.info("Retrieved %d evidence items.", len(results))
        await ctx.send_message(
            RetrievalResult(
                topic=request.topic,
                evidence=evidence,
                requirements=request.requirements,
            )
        )

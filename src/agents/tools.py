"""
Domain tools exposed to agents via the @tool decorator.

Each tool is a standalone async function that the Agent Framework
automatically converts into a function-tool schema for the LLM.
"""

import json
from typing import Annotated, Any

from agent_framework import tool
from pydantic import Field

from src.core.logging import get_logger
from src.models.cpg import CPGDocument
from src.models.search import SearchFilter, SearchQuery
from src.services.search_client import hybrid_search

logger = get_logger("tools")

# Lazy reference to avoid circular import — set by factory.py at init time
_cpg_workflow = None
_session_store_ref = None


# ── Knowledge Retrieval ─────────────────────────────────────────


@tool(
    name="search_clinical_evidence",
    description=(
        "Search the clinical knowledge base for evidence, literature, and "
        "guidelines relevant to a medical topic. Returns ranked results from "
        "Azure AI Search using hybrid (keyword + vector + semantic) retrieval."
    ),
)
async def search_clinical_evidence(
    query: Annotated[str, Field(description="Clinical search query, e.g. 'Type 2 diabetes management in elderly'")],
    specialty: Annotated[str | None, Field(description="Medical specialty filter, e.g. 'Cardiology', 'Endocrinology'")] = None,
    min_publication_year: Annotated[int | None, Field(description="Minimum publication year filter")] = None,
    evidence_level: Annotated[str | None, Field(description="Evidence level filter: High, Moderate, Low, Very Low")] = None,
    top_k: Annotated[int, Field(description="Number of results to return")] = 10,
) -> str:
    """Search clinical knowledge base for relevant evidence."""
    search_filter = SearchFilter(
        specialty=specialty,
        min_publication_year=min_publication_year,
        evidence_level=evidence_level,
    )
    search_query = SearchQuery(
        query_text=query,
        filter=search_filter if any([specialty, min_publication_year, evidence_level]) else None,
        top_k=top_k,
        use_semantic_ranking=True,
        use_vector_search=True,
    )

    results = await hybrid_search(search_query)
    logger.info("search_clinical_evidence returned %d results for: %s", len(results), query[:100])

    if not results:
        return "No relevant clinical evidence found for this query."

    formatted = []
    for r in results:
        formatted.append(
            f"[{r.title}] (Source: {r.source}, Year: {r.publication_year}, "
            f"Evidence: {r.evidence_level}, Score: {r.score:.2f})\n{r.content}"
        )
    return "\n\n---\n\n".join(formatted)


# ── CPG Document Management ────────────────────────────────────


CPG_SECTION_NAMES = [
    "title", "executive_summary", "scope_and_purpose", "target_population",
    "clinical_recommendations", "evidence_summary", "risk_assessment",
    "contraindications", "monitoring_and_followup", "references",
    "review_date", "authors", "specialty",
]


@tool(
    name="validate_cpg_structure",
    description=(
        "Validate that a CPG JSON document conforms to the mandatory template. "
        "Returns a list of structural issues found."
    ),
)
def validate_cpg_structure(
    cpg_json: Annotated[str, Field(description="The CPG document as a JSON string")],
) -> str:
    """Validate CPG template compliance."""
    try:
        data = json.loads(cpg_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    required_sections = [
        "title", "executive_summary", "scope_and_purpose", "target_population",
        "clinical_recommendations", "evidence_summary", "risk_assessment",
        "contraindications", "monitoring_and_followup", "references",
    ]

    issues: list[str] = []
    for section in required_sections:
        value = data.get(section)
        if value is None:
            issues.append(f"CRITICAL: Required section '{section}' is missing.")
        elif isinstance(value, str) and not value.strip():
            issues.append(f"MAJOR: Section '{section}' is empty.")
        elif isinstance(value, list) and len(value) == 0:
            issues.append(f"MAJOR: Section '{section}' has no items.")

    if not issues:
        return "All required sections are present and non-empty. Template is compliant."

    return "Template validation issues:\n" + "\n".join(f"- {i}" for i in issues)


@tool(
    name="store_cpg_document",
    description=(
        "Persist a CPG document in the session store. "
        "Call this after generating or modifying a CPG. "
        "Returns the CPG ID and version."
    ),
)
async def store_cpg_document(
    cpg_json: Annotated[str, Field(description="The complete CPG document as a JSON string")],
    **kwargs: Any,
) -> str:
    """Store a CPG document in the session store."""
    session_store = kwargs.get("session_store")
    if session_store is None:
        return "Error: session_store not available."

    try:
        data = json.loads(cpg_json)
        cpg = CPGDocument(**data)
    except Exception as exc:
        return f"Error parsing CPG document: {exc}"

    await session_store.save_cpg(cpg)
    return f"CPG stored successfully. ID: {cpg.id}, Version: {cpg.version}"


@tool(
    name="retrieve_cpg_document",
    description=(
        "Retrieve the current CPG document from the session store by its ID. "
        "Returns the full CPG JSON."
    ),
)
async def retrieve_cpg_document(
    cpg_id: Annotated[str, Field(description="The CPG document ID to retrieve")],
    **kwargs: Any,
) -> str:
    """Retrieve a CPG document from the session store."""
    session_store = kwargs.get("session_store")
    if session_store is None:
        return "Error: session_store not available."

    cpg = await session_store.get_cpg(cpg_id)
    if cpg is None:
        return f"No CPG document found with ID: {cpg_id}"

    return json.dumps(cpg.model_dump(mode="json"), indent=2)


# ── CPG Generation Pipeline (Workflow) ──────────────────────────


def set_workflow_refs(workflow, session_store) -> None:
    """Called by factory.py to inject the workflow and store references."""
    global _cpg_workflow, _session_store_ref
    _cpg_workflow = workflow
    _session_store_ref = session_store


@tool(
    name="run_cpg_pipeline",
    description=(
        "Run the full deterministic CPG generation pipeline for a NEW clinical "
        "topic. This executes a structured workflow: "
        "1) Search clinical evidence → 2) Author the CPG → 3) Review the CPG. "
        "Use this ONLY for creating a brand-new CPG. For modifying an existing "
        "CPG, use retrieve_cpg_document + generate_or_modify_cpg instead."
    ),
)
async def run_cpg_pipeline(
    topic: Annotated[str, Field(description="The clinical topic for the new CPG, e.g. 'Management of Type 2 Diabetes in elderly patients'")],
    specialty: Annotated[str | None, Field(description="Medical specialty, e.g. 'Endocrinology'")] = None,
    requirements: Annotated[str, Field(description="Additional requirements or constraints for the CPG")] = "",
    **kwargs: Any,
) -> str:
    """Execute the full CPG generation workflow and persist the result."""
    from src.agents.executors import RetrievalRequest

    if _cpg_workflow is None:
        return "Error: CPG generation workflow not initialised."

    logger.info("Running CPG pipeline for topic: %s", topic[:100])

    request = RetrievalRequest(
        topic=topic,
        specialty=specialty,
        requirements=requirements,
    )

    # Run the workflow — deterministic: Retrieval → Author → Reviewer
    collected_outputs: list[str] = []
    async for event in _cpg_workflow.run_stream(request):
        if event.type == "output" and event.data:
            collected_outputs.append(str(event.data))

    if not collected_outputs:
        return "Pipeline completed but produced no output."

    # The last output is the reviewer's assessment; the second-to-last is the authored CPG
    pipeline_result = "\n\n".join(collected_outputs)

    # Try to extract and persist the CPG JSON from the pipeline output
    session_store = kwargs.get("session_store") or _session_store_ref
    if session_store:
        for output in reversed(collected_outputs):
            try:
                # Find JSON in the output (the author agent outputs JSON)
                start = output.index("{")
                end = output.rindex("}") + 1
                cpg_json = output[start:end]
                cpg = CPGDocument(**json.loads(cpg_json))
                await session_store.save_cpg(cpg)
                logger.info("Pipeline CPG persisted: ID=%s, Version=%d", cpg.id, cpg.version)
                return (
                    f"CPG generated and stored successfully.\n"
                    f"CPG ID: {cpg.id}\n"
                    f"Version: {cpg.version}\n"
                    f"Title: {cpg.title}\n\n"
                    f"Full pipeline output:\n{pipeline_result}"
                )
            except (ValueError, json.JSONDecodeError):
                continue

    return f"Pipeline completed.\n\n{pipeline_result}"

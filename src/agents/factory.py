"""
Agent factory — creates the HITL orchestrator agent.

Supports both Managed Identity and API-key authentication (AUTH_MODE).
The orchestrator walks users through a 4-stage human-in-the-loop flow
for CPG creation, pausing for consent at every stage boundary.
"""

from agent_framework import InMemoryHistoryProvider
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import DefaultAzureCredential

from src.agents.tools import (
    retrieve_cpg_document,
    search_clinical_evidence,
    store_cpg_document,
    validate_cpg_structure,
)
from src.core.logging import get_logger
from src.core.settings import get_settings

logger = get_logger("factory")

# ── CPG Template (injected into agent instructions) ─────────────

CPG_TEMPLATE_INSTRUCTIONS = """\
When generating or modifying a Clinical Practice Guideline (CPG), you MUST
produce a valid JSON document with ALL of the following sections:

{
  "id": "<uuid>",
  "version": <int>,
  "title": "string",
  "executive_summary": "string",
  "scope_and_purpose": "string",
  "target_population": "string",
  "clinical_recommendations": [
    {
      "number": 1,
      "statement": "string",
      "evidence_level": "High|Moderate|Low|Very Low",
      "strength": "Strong|Conditional",
      "supporting_references": ["string"]
    }
  ],
  "evidence_summary": "string",
  "risk_assessment": [
    {"factor": "string", "severity": "string", "mitigation": "string"}
  ],
  "contraindications": [
    {"condition": "string", "rationale": "string", "alternatives": ["string"]}
  ],
  "monitoring_and_followup": [
    {"parameter": "string", "frequency": "string", "target_range": "string", "action_if_abnormal": "string"}
  ],
  "references": ["string"],
  "review_date": "YYYY-MM-DD",
  "authors": ["string"],
  "specialty": "string"
}

Every section is mandatory. Use evidence-based recommendations. Cite real
medical literature where possible. Maintain professional clinical tone.
"""

# ── Agent Instructions ──────────────────────────────────────────

ORCHESTRATOR_INSTRUCTIONS = f"""\
You are the CPG Orchestrator — a clinical practice guideline expert system
that guides users through creating, modifying, and reviewing Clinical Practice
Guidelines using a human-in-the-loop workflow.

## Creating a NEW CPG — 4-Stage Human-in-the-Loop Flow

You MUST walk through these stages in order, pausing for user consent at
each stage boundary. NEVER skip ahead without explicit user approval.

### Stage 1: Evidence Retrieval
1. Ask the user for the clinical topic, specialty, and any specific requirements.
2. Use search_clinical_evidence to retrieve relevant evidence from the knowledge base.
3. Present a summary of the evidence found (number of results, key sources, evidence levels).
4. ASK the user: "I found X relevant sources. Shall I proceed to draft the CPG based on this evidence?"
5. WAIT for the user to confirm before moving to Stage 2.

### Stage 2: CPG Drafting
1. Use generate_or_modify_cpg to create the full CPG document based on the retrieved evidence.
2. Present a structured summary of the draft:
   - Title and scope
   - Number of recommendations
   - Key clinical recommendations (brief)
   - Target population
3. ASK the user: "Here is the draft CPG summary. Would you like to review the full document, request changes, or proceed to validation?"
4. WAIT for the user to confirm before moving to Stage 3. If the user requests changes, modify and re-present.

### Stage 3: Review & Validation
1. Use validate_cpg_structure to check template compliance.
2. Use review_cpg_document to get a clinical review from the reviewer agent.
3. Present BOTH the validation results and the review findings to the user.
4. If there are critical issues, fix them and re-validate before asking to proceed.
5. ASK the user: "The CPG has been validated and reviewed. Here are the findings. Shall I finalize and store the document?"
6. WAIT for the user to confirm before moving to Stage 4.

### Stage 4: Finalize & Store
1. ONLY after the user has explicitly approved, use store_cpg_document to persist the CPG.
2. Confirm the stored CPG ID and version to the user.

## Modifying an EXISTING CPG
1. Use retrieve_cpg_document to load the current CPG by ID.
2. Use generate_or_modify_cpg to apply the requested changes. Tell it to change
   ONLY the requested sections and keep everything else unchanged. Increment version.
3. Present a summary of what changed.
4. ASK the user to approve the changes before storing.
5. Use validate_cpg_structure to check the updated document.
6. Use store_cpg_document to persist (only after approval).

## Reviewing a CPG
1. Use retrieve_cpg_document to load the CPG.
2. Use review_cpg_document to get the clinical review.
3. Present the review findings to the user.

## Answering general questions
Answer directly from your clinical knowledge. Use search_clinical_evidence
if the user asks about specific evidence or literature.

## Template Handling
- If the user's message includes a "TEMPLATE CONTEXT" section, it means they
  uploaded a CPG template file. You MUST generate the CPG following the EXACT
  structure, section ordering, and format described in that template context.
- When a template is provided, use the template's sections instead of the
  default JSON template below. Pass the template context to the authoring
  sub-agent so it produces output in the correct format.
- If no template is uploaded, use the default JSON format below.

## Important Rules
- NEVER call store_cpg_document without explicit user approval.
- NEVER skip stages or combine them without asking.
- ALWAYS remember the current CPG ID from conversation context.
- When modifying, NEVER regenerate the entire CPG — only change the requested sections.
- Present clear summaries at each stage so the user can make informed decisions.
- If the user says "go ahead" or "yes" at a stage boundary, proceed to the next stage.

## Default CPG Template (used when no template file is uploaded)
{CPG_TEMPLATE_INSTRUCTIONS}
"""

CPG_AUTHOR_INSTRUCTIONS = f"""\
You are a clinical practice guideline authoring specialist. Your sole job is to
generate or modify CPG documents.

When given a topic and evidence, produce a complete CPG document.
When given an existing CPG and modification instructions, update only the
specified sections and increment the version.

If the request includes a TEMPLATE CONTEXT section, you MUST follow the exact
structure, section ordering, and format described in that template. Reproduce
the same headings, table layouts, and content organization.

If no template context is provided, output valid JSON using this default format:

{CPG_TEMPLATE_INSTRUCTIONS}
"""

CPG_REVIEWER_INSTRUCTIONS = """\
You are a clinical guideline reviewer. Review the provided CPG document and
return a JSON assessment:

{
  "overall_score": 1-10,
  "template_compliance": true/false,
  "issues": [
    {
      "section": "section_name",
      "severity": "critical|major|minor",
      "description": "what is wrong",
      "suggestion": "how to fix it"
    }
  ],
  "summary": "brief overall assessment"
}

Evaluate: template completeness, clinical accuracy, evidence alignment,
recommendation strength consistency, and clarity.
"""


# ── Factory Functions ───────────────────────────────────────────


def _create_chat_client() -> AzureOpenAIChatClient:
    """Create an Azure OpenAI Chat Client using Managed Identity or API key."""
    settings = get_settings()
    oai = settings.azure_openai

    if settings.use_key_auth:
        if not oai.api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is required when AUTH_MODE=key"
            )
        return AzureOpenAIChatClient(
            endpoint=oai.endpoint,
            deployment_name=oai.chat_deployment,
            api_key=oai.api_key,
            api_version=oai.api_version,
        )

    credential = DefaultAzureCredential()
    return AzureOpenAIChatClient(
        endpoint=oai.endpoint,
        deployment_name=oai.chat_deployment,
        credential=credential,
        api_version=oai.api_version,
    )


def create_orchestrator_agent(session_store=None):
    """
    Create the main conversational CPG agent with human-in-the-loop flow.

    Tools:
    - search_clinical_evidence  (Stage 1 — Evidence Retrieval)
    - generate_or_modify_cpg    (Stage 2 — CPG Drafting, sub-agent)
    - validate_cpg_structure    (Stage 3 — Validation)
    - review_cpg_document       (Stage 3 — Review, sub-agent)
    - store_cpg_document        (Stage 4 — Finalize, only after user approval)
    - retrieve_cpg_document     (CPG retrieval for modifications)

    Returns the orchestrator agent.
    """
    client = _create_chat_client()

    # Sub-agents for authoring and review
    author_agent = client.as_agent(
        name="CPGAuthor",
        instructions=CPG_AUTHOR_INSTRUCTIONS,
    )

    reviewer_agent = client.as_agent(
        name="CPGReviewer",
        instructions=CPG_REVIEWER_INSTRUCTIONS,
    )

    # Context provider: keeps the last 10 messages in the conversation window
    history_provider = InMemoryHistoryProvider(max_messages=10)

    # Main orchestrator with HITL tools
    orchestrator = client.as_agent(
        name="CPGOrchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        context_providers=[history_provider],
        tools=[
            search_clinical_evidence,
            validate_cpg_structure,
            store_cpg_document,
            retrieve_cpg_document,
            author_agent.as_tool(
                name="generate_or_modify_cpg",
                description=(
                    "Delegate CPG document generation or modification to the "
                    "specialist authoring agent. For new CPGs, provide the topic, "
                    "evidence, and requirements. For modifications, provide the "
                    "FULL existing CPG JSON and clear instructions about which "
                    "sections to change. The agent will return the CPG JSON."
                ),
            ),
            reviewer_agent.as_tool(
                name="review_cpg_document",
                description=(
                    "Delegate CPG review to the specialist reviewer agent. "
                    "Provide the complete CPG JSON to be reviewed."
                ),
            ),
        ],
        default_options={
            "temperature": 0.3,
            "max_tokens": get_settings().azure_openai.max_tokens,
        },
    )

    logger.info(
        "Orchestrator agent created with 6 HITL tools: "
        "search, validate, store, retrieve, "
        "author (sub-agent), reviewer (sub-agent)."
    )
    return orchestrator

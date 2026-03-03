"""
Agent and Workflow factory.

Creates the Agent Framework agents, wires tools, and builds workflows.
All agents use Azure OpenAI via Managed Identity.

Architecture (hybrid):
- New CPG creation  → deterministic Workflow pipeline (retrieve → author → review)
- CPG modification  → conversational tool calls (orchestrator picks tools)
- CPG review        → conversational tool call to reviewer sub-agent
- General questions → orchestrator answers directly via AgentSession memory
"""

from agent_framework import InMemoryHistoryProvider, WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import DefaultAzureCredential

from src.agents.executors import KnowledgeRetrievalExecutor
from src.agents.tools import (
    retrieve_cpg_document,
    run_cpg_pipeline,
    search_clinical_evidence,
    set_workflow_refs,
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
You are the CPG Multi-Agent Orchestrator — a clinical practice guideline expert
system. You help users create, modify, and review Clinical Practice Guidelines.

## Creating a NEW CPG
Use the run_cpg_pipeline tool. It runs a deterministic pipeline:
  1. Searches clinical evidence from the knowledge base
  2. Generates the complete CPG via the specialist authoring agent
  3. Reviews the CPG via the specialist reviewer agent
  4. Persists the CPG and returns the ID
Just call run_cpg_pipeline with the topic, specialty, and any requirements.
After the pipeline completes, summarise the result and tell the user the CPG ID.

## Modifying an EXISTING CPG
Use conversational tools step by step:
  1. retrieve_cpg_document — load the current CPG by ID
  2. generate_or_modify_cpg — pass the existing CPG JSON and the modification
     instruction to the authoring sub-agent. Tell it to change ONLY the
     requested sections and keep everything else unchanged. Increment version.
  3. validate_cpg_structure — check the updated document
  4. store_cpg_document — persist the new version
  5. Tell the user what changed and the new version number

## Reviewing a CPG
  1. retrieve_cpg_document — load the CPG
  2. review_cpg_document — pass the CPG to the reviewer sub-agent
  3. Present the review findings to the user

## Answering general questions
Answer directly from your clinical knowledge. Use search_clinical_evidence
if the user asks about specific evidence or literature.

## Important rules
- ALWAYS remember the current CPG ID from conversation context so the user
  doesn't have to repeat it.
- When modifying, NEVER regenerate the entire CPG — only change the requested
  sections.
- After any CPG change, validate and store the result.

{CPG_TEMPLATE_INSTRUCTIONS}
"""

CPG_AUTHOR_INSTRUCTIONS = f"""\
You are a clinical practice guideline authoring specialist. Your sole job is to
generate or modify CPG documents in strict JSON format.

When given a topic and evidence, produce a complete CPG document.
When given an existing CPG and modification instructions, update only the
specified sections and increment the version.

Always output valid JSON and nothing else.

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
    """Create an Azure OpenAI Chat Client using Managed Identity."""
    settings = get_settings().azure_openai
    credential = DefaultAzureCredential()
    return AzureOpenAIChatClient(
        endpoint=settings.endpoint,
        deployment_name=settings.chat_deployment,
        credential=credential,
        api_version=settings.api_version,
    )


def _build_cpg_generation_workflow(client: AzureOpenAIChatClient):
    """
    Build the deterministic CPG generation pipeline:
      KnowledgeRetrievalExecutor → CPG Author Agent → CPG Reviewer Agent

    Returns the built Workflow object.
    """
    retrieval_executor = KnowledgeRetrievalExecutor()

    author_agent = client.as_agent(
        name="PipelineCPGAuthor",
        instructions=CPG_AUTHOR_INSTRUCTIONS,
    )

    reviewer_agent = client.as_agent(
        name="PipelineCPGReviewer",
        instructions=CPG_REVIEWER_INSTRUCTIONS,
    )

    workflow = (
        WorkflowBuilder(start_executor=retrieval_executor)
        .add_edge(retrieval_executor, author_agent)
        .add_edge(author_agent, reviewer_agent)
        .build()
    )

    logger.info("CPG generation workflow built: Retrieval → Author → Reviewer.")
    return workflow


def create_orchestrator_agent(session_store=None):
    """
    Create the main conversational CPG agent (hybrid architecture).

    - NEW CPG creation → delegates to run_cpg_pipeline (deterministic workflow)
    - CPG modification → uses generate_or_modify_cpg sub-agent (conversational)
    - CPG review       → uses review_cpg_document sub-agent (conversational)
    - Evidence search  → uses search_clinical_evidence tool
    - CPG persistence  → uses store/retrieve tools

    Returns the orchestrator agent.
    """
    client = _create_chat_client()

    # Build the deterministic workflow and inject it into the pipeline tool
    workflow = _build_cpg_generation_workflow(client)
    set_workflow_refs(workflow, session_store)

    # Sub-agents for conversational modification and review
    author_agent = client.as_agent(
        name="CPGAuthor",
        instructions=CPG_AUTHOR_INSTRUCTIONS,
    )

    reviewer_agent = client.as_agent(
        name="CPGReviewer",
        instructions=CPG_REVIEWER_INSTRUCTIONS,
    )

    # Context provider: keeps the last 10 messages in the conversation window.
    # Older messages are dropped to stay within the model's context budget.
    history_provider = InMemoryHistoryProvider(max_messages=10)

    # Main orchestrator with all tools
    orchestrator = client.as_agent(
        name="CPGOrchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        context_providers=[history_provider],
        tools=[
            # Deterministic pipeline for new CPG creation
            run_cpg_pipeline,
            # Conversational tools for modification / review
            search_clinical_evidence,
            validate_cpg_structure,
            store_cpg_document,
            retrieve_cpg_document,
            author_agent.as_tool(
                name="generate_or_modify_cpg",
                description=(
                    "Delegate CPG document generation or modification to the "
                    "specialist authoring agent. For modifications, provide the "
                    "FULL existing CPG JSON and clear instructions about which "
                    "sections to change. The agent will return the updated CPG JSON."
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
        "Orchestrator agent created with 7 tools: "
        "run_cpg_pipeline, search, validate, store, retrieve, "
        "author (sub-agent), reviewer (sub-agent)."
    )
    return orchestrator

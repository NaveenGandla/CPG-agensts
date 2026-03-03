# CPG Multi-Agent Healthcare System

Production-grade multi-agent backend for generating and refining **Clinical Practice Guidelines (CPG)** using Microsoft Agent Framework and Azure AI services.

Built on **Microsoft Agent Framework** (the successor to Semantic Kernel + AutoGen) for agent orchestration, tool calling, session memory, and graph-based workflows.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI (API Layer)                           │
│  POST /api/v1/chat  ·  GET /api/v1/cpg/{id}  ·  POST /api/v1/index  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Orchestrator Agent        │  ← AzureOpenAIChatClient.as_agent()
              │  (AgentSession for memory) │
              └──┬──────┬──────┬──────┬───┘
                 │      │      │      │
      ┌──────────┘      │      │      └──────────────┐
      ▼                 ▼      ▼                     ▼
┌───────────┐  ┌──────────────────┐  ┌──────────────────┐
│ @tool     │  │ Sub-Agent (tool) │  │ Sub-Agent (tool) │
│ search_   │  │ CPG Author       │  │ CPG Reviewer     │
│ clinical_ │  │ .as_tool()       │  │ .as_tool()       │
│ evidence  │  └────────┬─────────┘  └──────────────────┘
└─────┬─────┘           │
      │                 │                Additional @tools:
┌─────▼─────┐   ┌──────▼──────┐    ┌─────────────────────────┐
│ Azure AI  │   │ Azure OpenAI│    │ validate_cpg_structure   │
│ Search    │   │ (AI Foundry)│    │ store_cpg_document       │
│ (Hybrid + │   │ Chat + Embed│    │ retrieve_cpg_document    │
│ Semantic) │   │             │    └─────────────────────────┘
└───────────┘   └─────────────┘

    ┌──────────────────────────────────────────────────┐
    │  Optional: CPG Generation Workflow               │
    │  (WorkflowBuilder graph for batch pipelines)     │
    │                                                  │
    │  KnowledgeRetrievalExecutor → Author → Reviewer  │
    └──────────────────────────────────────────────────┘
```

### How Shared State & Communication Works

| Mechanism | Provided By | Purpose |
|-----------|-------------|---------|
| **AgentSession** | Agent Framework | Conversation memory across multi-turn interactions |
| **Agent-as-Tool** | Agent Framework | Orchestrator delegates to CPG Author / Reviewer sub-agents |
| **@tool functions** | Agent Framework | Domain tools (search, validate, store) called by agents |
| **Workflow edges** | Agent Framework | Type-safe message routing in the CPG generation pipeline |
| **Session Store** | Custom (Redis/memory) | CPG document persistence and version tracking |

### Agent Responsibilities

| Component | Type | Role |
|-----------|------|------|
| **Orchestrator** | Agent (with tools) | Conversational interface; decides which tools/sub-agents to invoke |
| **CPG Author** | Sub-Agent (as_tool) | Generates or modifies CPG documents in strict JSON template |
| **CPG Reviewer** | Sub-Agent (as_tool) | Reviews CPG for template compliance and clinical quality |
| **search_clinical_evidence** | @tool function | Hybrid + semantic search on Azure AI Search |
| **validate_cpg_structure** | @tool function | Structural validation of CPG template |
| **store/retrieve_cpg_document** | @tool functions | CPG persistence via session store |

---

## Azure Services Used

| Service | Purpose |
|---------|---------|
| **Azure OpenAI (AI Foundry)** | Chat completions (GPT-4o) and text embeddings |
| **Azure AI Search** | Vector + hybrid + semantic search over clinical knowledge base |
| **Azure Container Apps** | Serverless container hosting with auto-scaling |
| **Azure Container Registry** | Docker image storage |
| **Application Insights** | Structured logging, distributed tracing, metrics |
| **Azure Redis Cache** | CPG document persistence for multi-instance deployments (optional) |
| **Managed Identity** | Passwordless authentication to all Azure services |

---

## Project Structure

```
agents/
├── src/
│   ├── main.py                  # FastAPI app entry point
│   ├── agents/
│   │   ├── factory.py           # Creates agents, wires tools, builds workflows
│   │   ├── tools.py             # @tool functions (search, validate, store CPG)
│   │   └── executors.py         # Workflow Executors for CPG generation pipeline
│   ├── api/
│   │   ├── routes.py            # FastAPI endpoints (uses AgentSession)
│   │   └── schemas.py           # Request/response models
│   ├── core/
│   │   ├── settings.py          # Environment-based config (pydantic-settings)
│   │   ├── logging.py           # App Insights + OpenTelemetry
│   │   ├── auth.py              # Managed Identity (DefaultAzureCredential)
│   │   └── retry.py             # Exponential backoff for Azure calls
│   ├── models/
│   │   ├── cpg.py               # CPG document schema (Pydantic)
│   │   ├── session.py           # Session metadata (supplements AgentSession)
│   │   ├── search.py            # Search document/query models
│   │   └── agent.py             # Agent role enum (for logging)
│   └── services/
│       ├── openai_client.py     # Embedding endpoint only (chat via Agent Framework)
│       ├── search_client.py     # Azure AI Search (index schema + hybrid search)
│       └── session_store.py     # Redis / in-memory CPG document store
├── scripts/
│   ├── deploy.sh                # Azure CLI deployment
│   └── setup_search_index.py    # Index creation + sample data
├── tests/
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Azure subscription with:
  - Azure OpenAI resource (GPT-4o + text-embedding-3-small deployed)
  - Azure AI Search resource
  - Application Insights resource
- Azure CLI logged in (`az login`)

### 2. Local Development

```bash
cp .env.example .env
# Edit .env with your Azure resource values

pip install -r requirements.txt

# Create search index with sample data
python -m scripts.setup_search_index

# Run the API
uvicorn src.main:app --reload --port 8000
```

### 3. Usage Examples

**Create a CPG:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a clinical practice guideline for management of Type 2 Diabetes in elderly patients"}'
```

**Modify a section (same session):**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<SESSION_ID>", "message": "Change the target population to focus on patients over 75 years with renal impairment"}'
```

**Review the CPG:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<SESSION_ID>", "message": "Review the current guideline"}'
```

**Retrieve a CPG by ID:**
```bash
curl http://localhost:8000/api/v1/cpg/<CPG_ID>
```

---

## Deployment to Azure Container Apps

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

See [scripts/deploy.sh](scripts/deploy.sh) for the full CLI-based deployment.

---

## Required RBAC Role Assignments

Assign these roles to the Container App's system-assigned managed identity:

| Role | Scope | Purpose |
|------|-------|---------|
| `Cognitive Services OpenAI User` | Azure OpenAI resource | Chat completions + embeddings |
| `Search Index Data Contributor` | Azure AI Search resource | Read/write search index data |
| `Search Service Contributor` | Azure AI Search resource | Create/manage indexes |
| `Monitoring Metrics Publisher` | Application Insights resource | Telemetry ingestion |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | Embedding model deployment name |
| `AZURE_SEARCH_ENDPOINT` | Yes | Azure AI Search endpoint URL |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | No | App Insights connection string |
| `AZURE_REDIS_HOST` | No | Redis host for multi-instance sessions |
| `ENVIRONMENT` | No | `production` / `development` |
| `DEBUG` | No | Enable OpenAPI docs (`true`/`false`) |

See [.env.example](.env.example) for all options.

---

## Future Extension Strategy

The architecture makes adding new agents trivial with the Agent Framework:

### Option A: Add as a tool to the orchestrator

```python
# 1. Create the new agent
pathway_agent = client.as_agent(
    name="ClinicalPathwayAgent",
    instructions="Generate clinical pathways from CPG recommendations...",
)

# 2. Expose it as a tool on the orchestrator
orchestrator = client.as_agent(
    tools=[
        ...existing_tools,
        pathway_agent.as_tool(
            name="generate_clinical_pathway",
            description="Generate a clinical pathway from CPG recommendations",
        ),
    ],
)
```

### Option B: Add as a workflow step

```python
# Add a new executor to the CPG generation workflow
workflow = (
    WorkflowBuilder(start_executor=retrieval_executor)
    .add_edge(retrieval_executor, authoring_executor)
    .add_edge(authoring_executor, review_executor)
    .add_edge(review_executor, pathway_executor)  # new step
    .build()
)
```

### Option C: Add as a standalone @tool function

```python
@tool(name="generate_pathway", description="...")
async def generate_clinical_pathway(cpg_json: str) -> str:
    # Custom logic — no LLM needed
    ...
```

No orchestrator rewrite required in any case.

---

## Potential Enhancements

- **Azure API Management** — rate limiting, API key management, usage analytics
- **Azure Key Vault** — secret management instead of environment variables
- **Azure Storage** — persist CPG documents as blobs for long-term storage
- **Azure Event Grid** — event-driven notifications on CPG creation/updates
- **Streaming responses** — Agent Framework supports `run(..., stream=True)` natively
- **Human-in-the-loop** — Agent Framework's tool approval for clinical sign-off
- **Multi-language CPG** — Azure Translator integration
- **PDF export** — Generate downloadable CPG documents
- **Workflow checkpointing** — Agent Framework's built-in superstep checkpoints for long-running generation

"""
Azure AI Search client — vector, hybrid, and semantic search.
Includes index schema definition and document indexing.
"""

from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src.core.auth import get_credential
from src.core.logging import get_logger, get_tracer
from src.core.retry import async_retry
from src.core.settings import get_settings
from src.models.search import SearchDocument, SearchFilter, SearchQuery, SearchResult
from src.services.openai_client import generate_embeddings

logger = get_logger("search_client")
tracer = get_tracer("search_client")

VECTOR_DIMENSIONS = 1536  # text-embedding-ada-002 / text-embedding-3-small


def _build_index_schema(index_name: str) -> SearchIndex:
    """Define the AI Search index schema with vector, keyword, and semantic support."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(
            name="specialty", type=SearchFieldDataType.String, filterable=True
        ),
        SimpleField(
            name="publication_year",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="evidence_level", type=SearchFieldDataType.String, filterable=True
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(
            name="abstract", type=SearchFieldDataType.String
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="cpg-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="cpg-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="cpg-vector-profile", algorithm_configuration_name="cpg-hnsw"
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="cpg-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="abstract")],
        ),
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )


async def ensure_index_exists() -> None:
    """Create the search index if it does not already exist."""
    settings = get_settings().azure_search
    credential = get_credential()
    async with SearchIndexClient(
        endpoint=settings.endpoint, credential=credential
    ) as client:
        try:
            await client.get_index(settings.index_name)
            logger.info("Search index '%s' already exists.", settings.index_name)
        except Exception:
            index = _build_index_schema(settings.index_name)
            await client.create_index(index)
            logger.info("Created search index '%s'.", settings.index_name)


async def _get_search_client() -> SearchClient:
    settings = get_settings().azure_search
    credential = get_credential()
    return SearchClient(
        endpoint=settings.endpoint,
        index_name=settings.index_name,
        credential=credential,
    )


def _build_odata_filter(f: SearchFilter) -> str | None:
    """Build OData filter string from a SearchFilter."""
    clauses: list[str] = []
    if f.specialty:
        clauses.append(f"specialty eq '{f.specialty}'")
    if f.min_publication_year:
        clauses.append(f"publication_year ge {f.min_publication_year}")
    if f.max_publication_year:
        clauses.append(f"publication_year le {f.max_publication_year}")
    if f.evidence_level:
        clauses.append(f"evidence_level eq '{f.evidence_level}'")
    return " and ".join(clauses) if clauses else None


@async_retry()
async def hybrid_search(query: SearchQuery) -> list[SearchResult]:
    """Execute hybrid (keyword + vector) search with optional semantic ranking."""
    with tracer.start_as_current_span("hybrid_search") as span:
        span.set_attribute("query", query.query_text[:200])

        vector_queries = []
        if query.use_vector_search:
            embeddings = await generate_embeddings([query.query_text])
            vector_queries.append(
                VectorizedQuery(
                    vector=embeddings[0],
                    k_nearest_neighbors=query.top_k,
                    fields="content_vector",
                )
            )

        odata_filter = (
            _build_odata_filter(query.filter) if query.filter else None
        )

        async with await _get_search_client() as client:
            results = await client.search(
                search_text=query.query_text,
                vector_queries=vector_queries if vector_queries else None,
                filter=odata_filter,
                top=query.top_k,
                query_type="semantic" if query.use_semantic_ranking else "simple",
                semantic_configuration_name=(
                    get_settings().azure_search.semantic_config
                    if query.use_semantic_ranking
                    else None
                ),
                select=["id", "title", "content", "specialty", "publication_year",
                        "evidence_level", "source"],
            )

            search_results: list[SearchResult] = []
            async for r in results:
                search_results.append(
                    SearchResult(
                        id=r["id"],
                        title=r["title"],
                        content=r["content"],
                        score=r.get("@search.score", 0.0),
                        specialty=r.get("specialty", ""),
                        publication_year=r.get("publication_year", 0),
                        evidence_level=r.get("evidence_level", ""),
                        source=r.get("source", ""),
                        highlights=r.get("@search.highlights", {}).get("content", []),
                    )
                )

            span.set_attribute("result_count", len(search_results))
            return search_results


@async_retry()
async def index_documents(docs: list[SearchDocument]) -> int:
    """Index a batch of documents (with embeddings) into AI Search."""
    with tracer.start_as_current_span("index_documents") as span:
        # Generate embeddings for documents that don't have them
        texts = [d.content for d in docs if not d.content_vector]
        if texts:
            embeddings = await generate_embeddings(texts)
            idx = 0
            for d in docs:
                if not d.content_vector:
                    d.content_vector = embeddings[idx]
                    idx += 1

        async with await _get_search_client() as client:
            result = await client.upload_documents(
                documents=[d.model_dump() for d in docs]
            )
            succeeded = sum(1 for r in result if r.succeeded)
            span.set_attribute("indexed", succeeded)
            logger.info("Indexed %d / %d documents.", succeeded, len(docs))
            return succeeded

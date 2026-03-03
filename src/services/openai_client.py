"""
Azure OpenAI embedding client.

Chat completions are handled by the Microsoft Agent Framework agents.
This module only provides the embedding endpoint for Azure AI Search
vector generation.
"""

from openai import AsyncAzureOpenAI

from src.core.auth import get_credential
from src.core.logging import get_logger, get_tracer
from src.core.retry import async_retry
from src.core.settings import get_settings

logger = get_logger("openai_client")
tracer = get_tracer("openai_client")

_client: AsyncAzureOpenAI | None = None


async def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        settings = get_settings().azure_openai
        credential = get_credential()
        token = await credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        _client = AsyncAzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
            api_key=token.token,
        )
    return _client


@async_retry()
async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Azure OpenAI."""
    settings = get_settings().azure_openai
    client = await _get_client()

    with tracer.start_as_current_span("generate_embeddings") as span:
        span.set_attribute("batch_size", len(texts))
        response = await client.embeddings.create(
            model=settings.embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in response.data]

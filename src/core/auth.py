"""
Azure credential provider.

Supports two modes controlled by AUTH_MODE:
  - "identity": DefaultAzureCredential (Managed Identity / Azure CLI)
  - "key":      API keys supplied via environment variables

Use ``get_credential()`` for services that accept a TokenCredential.
For key-based authentication, individual service modules read keys
directly from their settings objects.
"""

from functools import lru_cache
from typing import Union

from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential

from src.core.settings import get_settings


@lru_cache
def get_credential() -> DefaultAzureCredential:
    """Return a shared async DefaultAzureCredential instance.

    Used when AUTH_MODE=identity.  Services that need key-based auth
    should read keys from their own settings instead.
    """
    return DefaultAzureCredential()


def get_search_credential() -> Union[DefaultAzureCredential, AzureKeyCredential]:
    """Return the appropriate credential for Azure AI Search."""
    settings = get_settings()
    if settings.use_key_auth:
        key = settings.azure_search.api_key
        if not key:
            raise ValueError(
                "AZURE_SEARCH_API_KEY is required when AUTH_MODE=key"
            )
        return AzureKeyCredential(key)
    return get_credential()


def get_doc_intelligence_credential() -> Union[DefaultAzureCredential, AzureKeyCredential]:
    """Return the appropriate credential for Azure Document Intelligence.

    Document Intelligence has always supported an optional API key;
    this helper respects both AUTH_MODE and the per-service AZURE_DI_KEY.
    """
    settings = get_settings()
    di = settings.azure_doc_intelligence
    # Per-service key takes precedence (backwards compatible)
    if di.api_key:
        return AzureKeyCredential(di.api_key)
    if settings.use_key_auth:
        raise ValueError(
            "AZURE_DI_KEY is required when AUTH_MODE=key"
        )
    return DefaultAzureCredential()

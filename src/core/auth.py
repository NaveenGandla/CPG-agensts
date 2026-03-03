"""
Azure Managed Identity credential provider.
Uses DefaultAzureCredential which chains:
  - Environment variables
  - Managed Identity (system / user-assigned)
  - Azure CLI (local dev)
"""

from functools import lru_cache

from azure.identity.aio import DefaultAzureCredential


@lru_cache
def get_credential() -> DefaultAzureCredential:
    """Return a shared async DefaultAzureCredential instance."""
    return DefaultAzureCredential()

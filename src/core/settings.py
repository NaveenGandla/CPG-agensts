"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI / AI Foundry deployment settings."""

    endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    api_version: str = Field("2024-12-01-preview", alias="AZURE_OPENAI_API_VERSION")
    chat_deployment: str = Field(..., alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    embedding_deployment: str = Field(..., alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    max_tokens: int = Field(4096, alias="AZURE_OPENAI_MAX_TOKENS")
    temperature: float = Field(0.3, alias="AZURE_OPENAI_TEMPERATURE")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AzureSearchSettings(BaseSettings):
    """Azure AI Search settings."""

    endpoint: str = Field(..., alias="AZURE_SEARCH_ENDPOINT")
    index_name: str = Field("cpg-knowledge-base", alias="AZURE_SEARCH_INDEX_NAME")
    semantic_config: str = Field(
        "cpg-semantic-config", alias="AZURE_SEARCH_SEMANTIC_CONFIG"
    )
    top_k: int = Field(10, alias="AZURE_SEARCH_TOP_K")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AzureDocIntelligenceSettings(BaseSettings):
    """Azure Document Intelligence for template parsing."""

    endpoint: Optional[str] = Field(None, alias="AZURE_DI_ENDPOINT")
    api_key: Optional[str] = Field(None, alias="AZURE_DI_KEY")
    model_id: str = Field("prebuilt-layout", alias="AZURE_DI_MODEL_ID")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AzureMonitorSettings(BaseSettings):
    """Application Insights / Azure Monitor settings."""

    connection_string: Optional[str] = Field(
        None, alias="APPLICATIONINSIGHTS_CONNECTION_STRING"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AzureRedisSettings(BaseSettings):
    """Azure Redis Cache for session state."""

    host: Optional[str] = Field(None, alias="AZURE_REDIS_HOST")
    port: int = Field(6380, alias="AZURE_REDIS_PORT")
    ssl: bool = Field(True, alias="AZURE_REDIS_SSL")
    session_ttl_seconds: int = Field(3600, alias="SESSION_TTL_SECONDS")

    model_config = {"env_prefix": "", "extra": "ignore"}


class AppSettings(BaseSettings):
    """Top-level application settings."""

    app_name: str = Field("cpg-multi-agent", alias="APP_NAME")
    environment: str = Field("production", alias="ENVIRONMENT")
    debug: bool = Field(False, alias="DEBUG")
    host: str = Field("0.0.0.0", alias="APP_HOST")
    port: int = Field(8000, alias="APP_PORT")

    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    azure_search: AzureSearchSettings = Field(default_factory=AzureSearchSettings)
    azure_doc_intelligence: AzureDocIntelligenceSettings = Field(
        default_factory=AzureDocIntelligenceSettings
    )
    azure_monitor: AzureMonitorSettings = Field(default_factory=AzureMonitorSettings)
    azure_redis: AzureRedisSettings = Field(default_factory=AzureRedisSettings)

    model_config = {"env_prefix": "", "extra": "ignore"}


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()

"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.logging import configure_logging, get_logger
from src.core.settings import get_settings
from src.services.search_client import ensure_index_exists

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    configure_logging()
    logger.info("Starting CPG Multi-Agent System...")

    # Ensure the Azure AI Search index exists
    try:
        await ensure_index_exists()
    except Exception as exc:
        logger.warning("Could not ensure search index: %s", exc)

    yield

    logger.info("Shutting down CPG Multi-Agent System.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CPG Multi-Agent Healthcare System",
        description=(
            "Production-grade multi-agent system for generating and refining "
            "Clinical Practice Guidelines using Azure AI services."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:5173"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()

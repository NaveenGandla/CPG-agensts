"""
Structured logging with Azure Application Insights integration.
Falls back to console logging when App Insights is not configured.
"""

import logging
import sys

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.core.settings import get_settings

_configured = False


def configure_logging() -> None:
    """Configure application logging with optional Azure Monitor integration."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.azure_monitor.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if settings.azure_monitor.connection_string:
        configure_azure_monitor(
            connection_string=settings.azure_monitor.connection_string,
            logger_name=settings.app_name,
        )
        logging.getLogger(settings.app_name).info(
            "Azure Monitor telemetry configured."
        )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{get_settings().app_name}.{name}")


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(f"{get_settings().app_name}.{name}")


def record_exception(span: trace.Span, exc: Exception) -> None:
    """Record an exception on the current OpenTelemetry span."""
    span.set_status(StatusCode.ERROR, str(exc))
    span.record_exception(exc)

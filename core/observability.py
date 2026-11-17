"""OpenInference-based observability bootstrap for the Financial AI Agent.

This module replaces the previous OpenLLMetry (Traceloop) instrumentation with
OpenInference, which is the approach Dynatrace recommends for AI Observability.

What it does:

* Builds an OpenTelemetry ``TracerProvider`` (the OpenInference subclass, which
  raises the per-span attribute limit so large prompts/completions are not
  truncated) and exports spans to Dynatrace over OTLP/HTTP.
* Auto-instruments LangChain **and LangGraph** through
  ``openinference-instrumentation-langchain``. Every agent, LLM call and tool
  invocation in the supervisor graph is captured without touching agent code.
* Enables the OpenTelemetry GenAI semantic conventions
  (``TraceConfig(enable_genai_semconv=True)``) so spans carry ``gen_ai.*``
  attributes alongside the native OpenInference ones. Dynatrace AI Observability
  needs ``gen_ai.*`` to light up model, token-usage and message-content views.
* Exposes :func:`get_tracer`, an OpenInference tracer that can create explicit
  ``chain`` / ``agent`` / ``tool`` spans and decorate functions.

If no Dynatrace token is configured the app still runs - tracing simply becomes
a no-op.
"""

from __future__ import annotations

import atexit
import os
from typing import Optional
from urllib.parse import urlparse

from core.logging_config import setup_logging

logger = setup_logging()

# Default service name, kept in sync with config.settings.AppConfig.service_name
DEFAULT_SERVICE_NAME = "FinancialAIAgent"

# OTLP/HTTP traces path that must be appended to the Dynatrace OTLP base URL.
_OTLP_TRACES_PATH = "/v1/traces"
# Dynatrace OTLP ingest base path, appended when only a tenant URL is given.
_DYNATRACE_OTLP_PATH = "/api/v2/otlp"

_initialized = False
_tracer = None


def _normalize_endpoint(raw_endpoint: str) -> str:
    """Turn whatever is in DYNATRACE_EXPORTER_OTLP_ENDPOINT into a traces URL.

    The OTLP/HTTP span exporter needs the full signal URL, while the Dynatrace
    docs, the devcontainer secret and Traceloop's ``api_endpoint`` all use the
    shorter base form. All of these are accepted:

    * ``abc12345.live.dynatrace.com``
    * ``https://abc12345.live.dynatrace.com``
    * ``https://abc12345.live.dynatrace.com/api/v2/otlp``
    * ``https://abc12345.live.dynatrace.com/api/v2/otlp/v1/traces``
    """
    endpoint = raw_endpoint.strip().rstrip("/")

    # Tolerate a bare host, which urlparse would otherwise read as a path.
    if not urlparse(endpoint).scheme:
        endpoint = f"https://{endpoint}"

    if endpoint.endswith(_OTLP_TRACES_PATH):
        return endpoint

    if not urlparse(endpoint).path.rstrip("/"):
        # Only a tenant URL was supplied - add the Dynatrace OTLP ingest path.
        endpoint = f"{endpoint}{_DYNATRACE_OTLP_PATH}"

    return f"{endpoint}{_OTLP_TRACES_PATH}"


def _auth_header(token: str) -> str:
    """Build the Authorization header value for a Dynatrace token.

    Platform tokens (``dt0s...``) use ``Bearer``; classic access tokens
    (``dt0c01...``) use ``Api-Token``.
    """
    return f"Bearer {token}" if token.startswith("dt0s") else f"Api-Token {token}"


def init_observability() -> None:
    """Configure OpenInference tracing. Safe to call more than once."""
    global _initialized, _tracer

    if _initialized:
        return
    _initialized = True

    # Imported lazily so the app still starts if the OTel extras are missing.
    from openinference.instrumentation import TraceConfig, TracerProvider
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME

    # Emit gen_ai.* attributes in addition to the OpenInference llm.* ones.
    # Set in code below via TraceConfig; mirrored into the environment so any
    # other OpenInference instrumentor in the process picks it up too.
    os.environ.setdefault("OPENINFERENCE_ENABLE_GENAI_SEMCONV", "true")
    config = TraceConfig(enable_genai_semconv=True)

    resource = Resource.create({SERVICE_NAME: service_name})
    tracer_provider = TracerProvider(resource=resource, config=config)

    endpoint = os.environ.get("DYNATRACE_EXPORTER_OTLP_ENDPOINT", "")
    token = os.environ.get("DYNATRACE_API_TOKEN", "")

    if endpoint and token:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        traces_endpoint = _normalize_endpoint(endpoint)
        exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers={"Authorization": _auth_header(token)},
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        atexit.register(tracer_provider.shutdown)
        logger.info("OpenInference tracing enabled, exporting to %s", traces_endpoint)
    else:
        logger.warning(
            "DYNATRACE_EXPORTER_OTLP_ENDPOINT / DYNATRACE_API_TOKEN not set - "
            "spans are created but not exported."
        )

    trace.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain + LangGraph (supervisor, agents, LLMs, tools).
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=config,
    )

    # Capture outbound HTTP made by the tools (yfinance, Tavily) as child spans.
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Requests instrumentation unavailable: %s", exc)

    _tracer = tracer_provider.get_tracer(__name__)


def get_tracer():
    """Return the OpenInference tracer used for explicit spans and decorators.

    The returned object is an ``OITracer``, so it supports both
    ``tracer.start_as_current_span(..., openinference_span_kind="chain")`` and
    the ``@tracer.chain`` / ``@tracer.agent`` / ``@tracer.tool`` decorators.
    """
    global _tracer

    if _tracer is None:
        init_observability()

    if _tracer is None:  # init failed - fall back to a tracer over the global provider
        from openinference.instrumentation import OITracer, TraceConfig
        from opentelemetry import trace

        _tracer = OITracer(trace.get_tracer(__name__), config=TraceConfig())

    return _tracer

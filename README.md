# FinancialAIAgent-Langchain
This sample application does stock analysis by conducting news search, fundamental review and technical analysis.
This app is built on a Supervisor with multiple agents architecture as described [here](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb).
The agents use Azure OpenAI, TavilySearch for news, and Yahoo Finance for fundamental and technical analysis.

### API Keys

#### Required
* `AZURE_ENDPOINT` — Azure OpenAI resource endpoint
* `AZURE_DEPLOYMENT` — Azure OpenAI deployment name
* `AZURE_SUBSCRIPTION_KEY` — Azure OpenAI API key
* `AZURE_API_VERSION` — Azure OpenAI API version (e.g. `2024-02-01`)
* `TAVILY_API_KEY` — Tavily search API key (used by the news agent)

#### Optional — Observability (Dynatrace via OpenInference)
* `DYNATRACE_EXPORTER_OTLP_ENDPOINT` — tenant URL, OTLP base, or full traces URL. All of
  `<tenant>.live.dynatrace.com`, `.../api/v2/otlp` and `.../api/v2/otlp/v1/traces` work.
* `DYNATRACE_API_TOKEN` — classic access token (`dt0c01...`) with scope `openTelemetryTrace.ingest`,
  or a platform token (`dt0s...`) with scope `openpipeline:traces:ingest`. The correct
  `Api-Token` / `Bearer` prefix is chosen automatically.
* `OTEL_SERVICE_NAME` — Service name shown in traces (defaults to `FinancialAIAgent`)

If the endpoint or token is missing the app still runs; tracing simply becomes a no-op.

### Requirements
* Python 3.10 or higher (required by `openinference-instrumentation-langchain`)

### How to run

#### On Desktop
```bash
cp .env.template .env
# Edit .env with your keys

python -m venv financialagent
source financialagent/bin/activate
pip install -r requirements.txt

streamlit run main.py
```

#### Codespaces
The devcontainer fetches Azure and OTel secrets from a secret server at startup using a password you provide. Set the following as a Codespaces secret in your GitHub repository settings:
* `WORKSHOP_PASSWORD` — password for the secret server

Dynatrace and Tavily keys must also be set as Codespaces secrets:
* `DYNATRACE_EXPORTER_OTLP_ENDPOINT`
* `DYNATRACE_API_TOKEN`
* `TAVILY_API_KEY`

Once secrets are configured, open the repo in Codespaces and run:
```bash
./run.sh
```

### Observability

The app is instrumented with **[OpenInference](https://github.com/Arize-ai/openinference)**, the
approach [Dynatrace recommends](https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/get-started/openinference)
for AI Observability. It replaces the previous OpenLLMetry / Traceloop setup, which Dynatrace
deprecates on 2027-01-31.

All tracing setup lives in [`core/observability.py`](core/observability.py):

* `openinference-instrumentation-langchain` hooks into LangChain's callback system, so the
  **whole LangGraph supervisor graph** is traced automatically — the supervisor, each of the
  three agents, every Azure OpenAI call and every tool invocation. No decorators on agent code.
* `TraceConfig(enable_genai_semconv=True)` makes spans carry OpenTelemetry `gen_ai.*`
  attributes alongside the native OpenInference `llm.*` ones. Dynatrace AI Observability needs
  `gen_ai.*` for its model, token-usage and message-content views.
* Spans are exported over OTLP/HTTP straight to Dynatrace — no collector required.
* Each user query gets an explicit root `chain` span named `financial_analysis` carrying the
  query as input and the final answer as output, tagged with `session.id` /
  `gen_ai.conversation.id` so a full chat conversation can be followed in Dynatrace.
* Outbound HTTP from the tools (Yahoo Finance, Tavily) is captured via
  `opentelemetry-instrumentation-requests`.

#### Adding spans to non-LangChain code

LangChain and LangGraph code needs no decorators. For plain Python helpers, use the
OpenInference tracer:

```python
from core.observability import get_tracer

tracer = get_tracer()

@tracer.agent(name="my_agent")      # also: tracer.chain, tracer.tool, tracer.llm
def my_helper(symbol: str) -> str:
    ...
```

#### Controlling captured content

Prompt and completion content is captured by default. To suppress it, set any of
`OPENINFERENCE_HIDE_INPUTS`, `OPENINFERENCE_HIDE_OUTPUTS`, `OPENINFERENCE_HIDE_INPUT_MESSAGES`
or `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` to `true`.

#### Note on GenAI metrics

OpenInference is span-only — it emits no metric instruments, so the
`gen_ai.client.token.usage` and `gen_ai.client.operation.duration` metrics are not produced by
the app. Token counts are still on the spans. To chart them as metrics, route the app through an
OpenTelemetry collector that derives them, as in the
[official Dynatrace LangGraph example](https://github.com/dynatrace-oss/dynatrace-ai-agent-instrumentation-examples/tree/main/langgraph/openinference).

### Suggested Chat Commands
* Analyze AAPL stock
* Give me latest news on Nvidia
* Conduct fundamental analysis on DIS
* What are the technical indicators for MSFT?

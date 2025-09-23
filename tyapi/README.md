# Tygent SaaS Planner (tyapi)

This folder contains a minimal SaaS-style service that ingests plans produced by any agent framework, enriches them with scheduling metadata, and returns a Tygent-native representation ready for accelerated execution.

## Features
- `POST /v1/plan/convert` endpoint that ingests framework plans and returns the canonical Tygent execution plan with dependency analysis and prefetch directives.
- Pluggable ingestors (`generic`, `langchain`, `crewai`, extendable) translate framework-specific payloads into the canonical schema.
- API-key enforcement via a lightweight tenant registry and CLI.
- Prefetch metadata aggregated from step prompts/links so the runtime can warm caches or download external resources before execution.

## Quickstart (Local)
1. Ensure dependencies are installed (`pip install -e .[dev]` if available).
2. Create an account and API key:
   ```bash
   python -m tyapi.service.cli register --name "Acme" --email "ops@acme.test"
   python -m tyapi.service.cli configure-ingestor --account <account_id> --name generic
   python -m tyapi.service.cli generate-key --account <account_id>
   ```
   The commands store metadata in `tyapi/service_state.json` by default. Set `TYGENT_SERVICE_STATE` to override the location.
3. Run the HTTP service:
   ```bash
   python -m tyapi.service.cli serve --port 8080
   ```
4. Call the conversion endpoint:
   ```bash
   curl -X POST http://127.0.0.1:8080/v1/plan/convert \
     -H "Content-Type: application/json" \
     -H "X-Tygent-Key: <api_key>" \
     -d '{
           "ingestor": {"name": "generic"},
           "plan": {
             "steps": [
               {"name": "discover", "prompt": "Collect sources", "deps": []},
               {"name": "analyze", "prompt": "Synthesize", "deps": ["discover"], "links": ["https://docs"]}
             ]
           },
           "options": {"redundancy_mode": "inline"}
         }'
   ```

## Deploying to the Cloud
- Package the application (e.g. `uvicorn tyapi.service.app:create_app --factory`).
- Mount a persistent volume for the service state file or back it with a managed database (swap out `ServiceState`).
- Configure environment variables for provider credentials (OpenAI, Google Generative AI, etc.) and set `TYGENT_SERVICE_STATE` to the mounted path.
- Place the service behind TLS and run behind an API gateway or load balancer. Scale with ASGI workers if using `uvicorn`/`gunicorn`.

## Tests
Run the tyapi specific tests with:
```bash
pytest tyapi/tests -q
```

## Extending Providers or Frameworks
Add new ingestor classes in `tyapi/service/ingestors.py` and register them through `PlanIngestorRegistry.register`. They automatically appear in the CLI `catalog` output. Omitting additional configuration yields a raw Tygent plan that can be executed by the runtime after calling the service.

## Consuming Plans Inside Tygent
- Use `tygent.service_bridge.ServicePlanBuilder` to convert the JSON payload into executable plan dictionaries.
- Call `tygent.accelerate` with the service plan directly—the bridge is invoked automatically when functions are missing.
- Prefetch hints are surfaced via `tygent.prefetch.prefetch_many` (default implementation is a stub—replace it with an HTTP cache for production).

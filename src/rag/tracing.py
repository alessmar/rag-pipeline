import os


def setup() -> None:
    """Send traces (OpenAI calls auto-instrumented, plus our own spans) to a local Phoenix server."""
    if os.environ.get("RAG_TRACING", "1") == "0":
        return
    from phoenix.otel import register

    register(
        project_name="rag",
        endpoint=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"),
        auto_instrument=True,
        batch=True,
        verbose=False,
    )

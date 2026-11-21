import json

from opentelemetry import trace

from rag.embed import embed_query
from rag.store import query as store_query

tracer = trace.get_tracer("rag")


def search(text: str, tenant: str, k: int = 8) -> list[dict]:
    """Embed the question, then let the store run its (vector plus keyword) search for this tenant."""
    with tracer.start_as_current_span("retrieve") as span:
        out = store_query(tenant, text, embed_query(text), k)
        # these attribute names make Phoenix render the span as a retriever with its documents
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("input.value", text)
        span.set_attribute("metadata", json.dumps({"tenant": tenant}))  # lets you filter traces by tenant in Phoenix
        for i, c in enumerate(out):
            span.set_attribute(f"retrieval.documents.{i}.document.id", c["id"])
            span.set_attribute(f"retrieval.documents.{i}.document.content", c["text"])
            span.set_attribute(f"retrieval.documents.{i}.document.score", c["score"])
        return out

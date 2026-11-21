"""Storage entry point. Everything database-specific sits behind the VectorStore interface below,
so a new backend is one new class plus one line in backend()."""
import json
import os
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Protocol

PDF_DIR = os.environ.get("RAG_PDF_DIR", "data/pdfs")  # one sub-folder per tenant
TENANTS_FILE = os.environ.get("RAG_TENANTS_FILE", "tenants.json")  # {"<uuid>": "label"}
BACKEND = os.environ.get("RAG_VECTOR_STORE", "lancedb")
DEFAULT_TENANT = os.environ.get("RAG_DEFAULT_TENANT")  # the company the dropdown opens on


class VectorStore(Protocol):
    """Chunk rows are dicts with: id, doc, path, page, heading, text, items, vector.
    Every method takes a tenant UUID and must never return another tenant's data."""

    def replace_doc(self, tenant: str, doc: str, rows: list[dict]) -> None:
        """Delete the tenant's existing chunks of this PDF, then add the new rows."""

    def search(self, tenant: str, text: str, vector: list[float], k: int) -> list[dict]:
        """Top k chunks for the question, best first. Each result has the row fields (without the vector)
        plus `score`. The score is backend-specific: use it for ordering only. Raise LookupError if the
        tenant has no documents."""

    def docs(self, tenant: str) -> Counter:
        """PDF name -> number of chunks. Empty for a tenant with no data."""

    def tenants(self) -> list[str]:
        """UUIDs of the tenants that hold data."""

    def reindex(self, tenant: str) -> Counter:
        """Rebuild the tenant's search indexes (and upgrade an older table). Returns how many chunks were
        detected as Italian, English or unsure. A backend with nothing to rebuild returns an empty Counter."""


@lru_cache
def backend() -> VectorStore:
    if BACKEND == "lancedb":
        from rag.lancedb_store import LanceStore

        return LanceStore()
    raise ValueError(f"unknown RAG_VECTOR_STORE {BACKEND!r}")


def tenant_id(value: str) -> str:
    """Canonical UUID string. Anything else raises ValueError, so a tenant can never point outside its own folder."""
    return str(uuid.UUID(str(value)))


def labels() -> dict[str, str]:
    """UUID -> display name, read from TENANTS_FILE. A malformed UUID in the file raises, so a typo is not ignored."""
    p = Path(TENANTS_FILE)
    return {tenant_id(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def default_tenant() -> str | None:
    """The configured default company as a canonical UUID. A malformed value raises, so a typo is not ignored."""
    return tenant_id(DEFAULT_TENANT) if DEFAULT_TENANT else None


def tenants() -> list[str]:
    return sorted(set(backend().tenants()) | set(labels()))  # a labelled tenant is listed before it has any document


def pdf_dir(tenant: str) -> Path:
    return Path(PDF_DIR) / tenant_id(tenant)


def replace_doc(tenant: str, doc: str, rows: list[dict]) -> None:
    backend().replace_doc(tenant_id(tenant), doc, rows)


def query(tenant: str, text: str, vector: list[float], k: int) -> list[dict]:
    return backend().search(tenant_id(tenant), text, vector, k)


def reindex(tenant: str) -> Counter:
    return backend().reindex(tenant_id(tenant))


def ingested_docs(tenant: str) -> Counter:
    return backend().docs(tenant_id(tenant))

"""HTTP front door for the queue: POST /jobs with {"tenant", "url"} and an x-api-key header. Redis is never exposed.
The payload is validated here; the worker still checks the URL itself before it downloads anything."""
import os
import secrets
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.jobs import submit_url
from rag.store import tenants

API_KEY = os.environ.get("RAG_API_KEY", "")
if len(API_KEY) < 16:
    raise RuntimeError("set RAG_API_KEY (at least 16 characters) to start the API")

app = FastAPI(title="rag ingestion queue", docs_url=None, redoc_url=None, openapi_url=None)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant: UUID
    url: str = Field(max_length=2048)

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        u = urlsplit(v)
        if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
            raise ValueError("must be an http(s) URL without credentials")
        return v


def _key(x_api_key: str = Header("")) -> None:
    if not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, "invalid or missing x-api-key")


@app.post("/jobs", status_code=202, dependencies=[Depends(_key)])
def create_job(job: Job) -> dict:
    tenant = str(job.tenant)
    if tenant not in tenants():
        raise HTTPException(422, "unknown tenant: add it to tenants.json first")
    try:
        return {"job_id": submit_url(tenant, job.url)}
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(503, "the queue is unavailable")

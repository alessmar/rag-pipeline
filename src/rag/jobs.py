"""Ingestion jobs: the way to queue one (a Celery task on Redis) and a small status table the UI reads."""
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from rag.fetch import doc_name
from rag.store import tenant_id

JOBS_DB = os.environ.get("RAG_JOBS_DB", "data/jobs.sqlite")
ACTIVE = ("queued", "running", "retrying", "cancelling")  # not finished: the list keeps refreshing
FINISHED = ("done", "failed", "cancelled")  # these can be removed from the list


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    Path(JOBS_DB).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(JOBS_DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute(
        "create table if not exists jobs (id text primary key, tenant text, doc text, state text,"
        " detail text default '', created text, updated text)"
    )
    return c


def set_state(job_id: str, state: str, detail: str = "") -> None:
    with closing(_conn()) as c, c:
        c.execute("update jobs set state = ?, detail = ?, updated = ? where id = ?", (state, detail[:500], _now(), job_id))


def start_job(job_id: str) -> bool:
    """Mark a job running, unless it was cancelled from the list while it waited: then it becomes cancelled and
    False comes back, so the worker drops it. One statement, so a cancel and a start can never both win."""
    with closing(_conn()) as c, c:
        cur = c.execute("update jobs set state = 'running', detail = '', updated = ? where id = ? and state != 'cancelling'",
                        (_now(), job_id))
        if cur.rowcount:
            return True
        c.execute("update jobs set state = 'cancelled', detail = 'cancelled before it started', updated = ? where id = ?",
                  (_now(), job_id))
        return False


def cancel_job(job_id: str, tenant: str) -> bool:
    """Ask for a job that has not started to be skipped. A running job cannot be stopped, so it returns False.
    The state is 'cancelling' until the worker reaches the job: removing the line sooner would let the job run."""
    with closing(_conn()) as c, c:
        cur = c.execute("update jobs set state = 'cancelling', detail = 'the worker will skip it', updated = ? "
                        "where id = ? and tenant = ? and state = 'queued'", (_now(), job_id, tenant_id(tenant)))
        return bool(cur.rowcount)


def delete_job(job_id: str, tenant: str) -> bool:
    """Remove a finished job from the list. A job still in progress (or another tenant's) is left alone."""
    with closing(_conn()) as c, c:
        cur = c.execute("delete from jobs where id = ? and tenant = ? and state in ('done', 'failed', 'cancelled')",
                        (job_id, tenant_id(tenant)))
        return bool(cur.rowcount)


def jobs_for(tenant: str, limit: int = 10) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute("select * from jobs where tenant = ? order by created desc limit ?", (tenant_id(tenant), limit))
        return [dict(r) for r in rows]


def _create(tenant: str, doc: str) -> str:
    job_id = uuid.uuid4().hex
    with closing(_conn()) as c, c:
        c.execute("insert into jobs (id, tenant, doc, state, created, updated) values (?, ?, ?, 'queued', ?, ?)",
                  (job_id, tenant, doc, _now(), _now()))
    return job_id


def submit_url(tenant: str, url: str) -> str:
    """Queue the ingestion of a PDF the worker will download from this URL. The URL goes into the message as given.
    Everything else (address, size, content) is checked by the worker, which is the one that connects."""
    tenant = tenant_id(tenant)
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError("the URL must start with https://")
    job_id = _create(tenant, doc_name(url))
    try:
        from rag.celery_app import ingest_pdf  # imported here so the Streamlit app does not need Celery

        ingest_pdf.apply_async(kwargs={"tenant": tenant, "url": url}, task_id=job_id)
    except Exception as e:
        set_state(job_id, "failed", f"could not queue: {type(e).__name__}: {e}")
        raise
    return job_id


def ensure_job(job: dict) -> bool:
    """Make sure a status row exists for a message. A producer on another machine creates none.
    Returns False when the message is not a usable job."""
    try:
        tenant = tenant_id(job["tenant"])
        if not isinstance(job["url"], str):  # a message is JSON, so it can hold anything
            raise TypeError("url must be a string")
        doc = doc_name(job["url"])
        job_id = str(job["job_id"])
    except (KeyError, TypeError, ValueError):
        return False
    with closing(_conn()) as c, c:
        c.execute("insert or ignore into jobs (id, tenant, doc, state, created, updated) values (?, ?, ?, 'queued', ?, ?)",
                  (job_id, tenant, doc, _now(), _now()))
    return True

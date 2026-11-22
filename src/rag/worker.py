"""What the worker does with one job: {"job_id": ..., "tenant": "<uuid>", "url": "https://..."}.
The PDF is downloaded here, from the URL, and then ingested. It is kept only if ingestion worked."""
import logging
import os
import uuid

from rag import jobs
from rag.fetch import Refused, doc_name, fetch_pdf
from rag.store import pdf_dir, tenant_id

log = logging.getLogger("rag.worker")


def process(job: dict, final: bool = True) -> None:
    """Download and ingest the PDF of this job. A job that can never succeed (bad message, blocked URL,
    not a PDF, PDF without text) is marked failed and returns normally, so the queue drops it. Any other error
    is re-raised so the queue retries; `final` says whether this was the last allowed attempt."""
    from rag.ingest import NoTextError, ingest  # heavy imports, only the worker needs them

    job_id = str(job.get("job_id") or "")
    if not jobs.ensure_job(job):  # also covers a message that is not a job at all
        log.warning("dropping a message that is not a usable job: %.200s", job)
        jobs.set_state(job_id, "failed", "invalid job")  # does nothing when there is no row
        return
    if not jobs.start_job(job_id):  # it was cancelled from the list while it waited
        log.info("job %s was cancelled before it started", job_id)
        return
    try:
        tenant = tenant_id(job["tenant"])  # the message is untrusted input
        target = pdf_dir(tenant) / doc_name(job["url"])
    except (KeyError, TypeError, ValueError) as e:
        jobs.set_state(job_id, "failed", f"invalid job: {e}")
        return
    # The PDF is downloaded to a hidden staging file and moves to its real name only after ingestion worked.
    # So a job that fails leaves no PDF behind, and never overwrites the PDF of a document ingested earlier.
    staging = target.with_name(f".staging-{uuid.uuid4().hex}.pdf")
    try:
        fetch_pdf(job["url"], staging)
        chunks = ingest(staging, tenant, target)
        os.replace(staging, target)
    except (Refused, NoTextError) as e:  # these can never succeed: marked failed, and the queue drops the job
        jobs.set_state(job_id, "failed", str(e))
        return
    except Exception as e:  # a download or an ingestion that may work later (timeout, HTTP 5xx)
        jobs.set_state(job_id, "failed" if final else "retrying", f"{type(e).__name__}: {e}")
        raise
    else:
        jobs.set_state(job_id, "done", f"{chunks} chunks")
    finally:
        staging.unlink(missing_ok=True)

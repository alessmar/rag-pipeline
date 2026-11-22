"""Celery app. Redis is both the broker (the queue) and the result backend. Start the worker with `rag worker`.

A message is the task `rag.ingest` with two arguments, tenant (a UUID) and url (a PDF the worker downloads)."""
import os
from urllib.parse import urlsplit

from celery import Celery

from rag.worker import process

REDIS_URL = os.environ.get("RAG_REDIS_URL", "redis://localhost:6379/0")
ATTEMPTS = int(os.environ.get("RAG_JOB_ATTEMPTS", "3"))  # in total, the first run included



def results_url(broker_url: str, explicit: str | None = None) -> str:
    """Job results go to database 1 of the same Redis server unless RAG_REDIS_RESULTS_URL says otherwise, so the queue
    (database 0) and the results can be cleared separately. Redis Cluster has only database 0: set the variable there."""
    return explicit or urlsplit(broker_url)._replace(path="/1").geturl()


RESULTS_URL = results_url(REDIS_URL, os.environ.get("RAG_REDIS_RESULTS_URL"))

app = Celery("rag", broker=REDIS_URL, backend=RESULTS_URL)
app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],  # never pickle: it can run code
    task_acks_late=True,  # a job is acknowledged after it ran, so a worker that dies hands it to another one
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # take one job at a time, never hoard
    task_track_started=True,
    result_expires=86400,
    # Redis redelivers a job that was not acknowledged within this time, so keep it above your slowest PDF
    broker_transport_options={"visibility_timeout": int(os.environ.get("RAG_VISIBILITY_SECONDS", "3600"))},
    broker_connection_retry_on_startup=True,
)


@app.task(name="rag.ingest", bind=True, autoretry_for=(Exception,), retry_backoff=60, retry_backoff_max=600,
          max_retries=ATTEMPTS - 1)
def ingest_pdf(self, tenant: str, url: str) -> None:
    # process() returns for a job that can never succeed (it is marked failed) and raises for one worth retrying.
    # The job id is the Celery task id, which a producer may choose.
    process({"job_id": self.request.id, "tenant": tenant, "url": url}, final=self.request.retries >= self.max_retries)

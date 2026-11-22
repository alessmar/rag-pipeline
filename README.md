# RAG pipeline

Question answering over PDF documents, with answers that cite the exact paragraph and page. PDFs are parsed with Docling, embedded locally, stored in LanceDB, and answered by an OpenAI model. Traces go to Phoenix.

![An answer with its sources, from a 12-page PDF](docs/answer.png)

## Why

I wanted to build a RAG pipeline that is performant and reusable, and to learn the pieces of RAG engineering along the way: parsing, chunking, embeddings, hybrid search, citations, and the queue that feeds them.

## How

I built it with the help of Claude Code, but I directed the architecture and chose the libraries myself, such as Docling for parsing and Celery for the ingestion queue.

## Quick start (Docker)

You need [Docker](https://docs.docker.com/get-docker/) with Compose, an [OpenAI API key](https://platform.openai.com/api-keys) and about 8 GB of free RAM.

```
git clone <this repository>
cd rag-pipeline
cp .env.example .env
```

Open `.env` and paste your key after `OPENAI_API_KEY=`. Then:

```
docker compose up --build
```

The first start builds the image (about 5 GB of disk) and the first PDF downloads the models (about 1.4 GB, once). When it is running:

- the chat app is at http://localhost:8501
- Phoenix, which shows what each question retrieved and sent to OpenAI, is at http://localhost:6006

**Add a PDF.** Documents arrive through the queue, from a URL. In a second terminal, give the app a public `https` link to a PDF:

```
docker compose exec app python scripts/produce.py --tenant "$RAG_DEFAULT_TENANT" https://example.com/yourfile.pdf
```

The job shows up in the sidebar, and the worker downloads and ingests it. To use a server on your own network, see `RAG_FETCH_ALLOW` below.

It takes about a minute per 30 pages. Reload the app and ask a question about it.

**Stop and clean up.** `Ctrl+C` stops everything, and your documents are kept. `docker compose down -v` also deletes the documents, the queue and the downloaded models.

Below you will find how to run it without Docker, how search works, how much RAM it needs, and how the ingestion queue behaves.

## Running it without Docker

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), Docker (for Redis), and an OpenAI API key. About 8 GB of RAM is a sensible minimum, see the memory section below.

**1. Install the dependencies**

```
uv sync
```

**2. Create `.env`** in the project folder, and keep it out of version control:

```
OPENAI_API_KEY=sk-...
RAG_REDIS_URL=redis://localhost:6379/0
```

Every other setting is optional and has a default (model, reasoning effort, retries, size limits). The ones used by the ingestion queue are listed in the section further down.

**3. Create `tenants.json`**, which lists the companies. Each company is a UUID with a display name. Skip this step if you already have the file, because this command would replace it:

```
python3 -c "import uuid, json; print(json.dumps({str(uuid.uuid4()): 'mycompany'}, indent=2))" > tenants.json
cat tenants.json
```

Note the UUID: every command and URL below uses it.

**4. Start Redis in Docker**, with a named volume so the queue survives restarts and removing the container:

```
docker volume create rag-redis-data
docker run -d --name rag-redis -p 127.0.0.1:6379:6379 -v rag-redis-data:/data \
  redis:7-alpine redis-server --appendonly yes --maxmemory-policy noeviction
```

`rag-redis-data` is a named volume: Docker manages it, and you can list it with `docker volume ls`. This publishes Redis on `localhost` only, writes its data to the volume (`--appendonly yes`), and stops it from ever discarding queued jobs to free memory (`--maxmemory-policy noeviction`). Later, `docker stop rag-redis` and `docker start rag-redis` stop and start it again, and the queue is kept.

**5. Start the worker** in its own terminal. It takes the queued jobs one at a time. The first run downloads the Docling and embedding models, which takes a few minutes.

```
uv run rag worker
```

**6. Start the chat app** in another terminal:

```
uv run streamlit run src/rag/app.py
```

Open `http://localhost:8501/?tenant=<your uuid>`. Without the `?tenant=` part the sidebar shows a dropdown of the companies in `tenants.json`, opened on the company set in `RAG_DEFAULT_TENANT` (a UUID in `.env`) or else the first one. You can also ask from the command line: `uv run rag ask --tenant <uuid> "your question"`.

**7. Optional: traces.** Phoenix shows what each question retrieved and sent to OpenAI. Run it in a third terminal, and open `http://localhost:6006`:

```
PHOENIX_WORKING_DIR=$PWD/data/phoenix uv run phoenix serve
```

If you do not run Phoenix, put `RAG_TRACING=0` in `.env`.

**8. Add a document.** There are two ways.

- **Directly, from a file on your machine.** It does not need Redis or the worker, and it blocks until the PDF is parsed:

  ```
  uv run rag ingest --tenant <uuid> path/to/file.pdf
  ```

- **Through the queue, from a URL**, the same way a real producer would:

  ```
  uv run python scripts/produce.py --tenant <uuid> https://example.com/file.pdf
  ```

  The worker only downloads public `https` URLs. To try the queue with a PDF from your own machine, serve its folder over HTTP and list `127.0.0.1` in `RAG_FETCH_ALLOW` (see below). Add `RAG_FETCH_ALLOW=127.0.0.1` to `.env`, restart the worker, then:

  ```
  python3 -m http.server 8000 --bind 127.0.0.1 --directory path/to/pdfs
  uv run python scripts/produce.py --tenant <uuid> http://127.0.0.1:8000/file.pdf
  ```

The job list in the app's sidebar shows each job as queued, running, done or failed, and the PDF count updates when it finishes. Each line has an X icon: on a finished job it removes the job from the list, and on a job that has not started yet it cancels it (the worker then skips it). A job in progress has no X, because a running ingestion cannot be stopped safely.

**Tests.** `uv run pytest` runs the suite. It needs neither Redis nor an OpenAI key, because it runs queued jobs in the calling process and replaces the OpenAI call.

**Stopping and cleaning up.** Stop the three processes with `Ctrl+C`. `docker stop rag-redis` stops Redis. To delete the queue completely, run `docker rm -f rag-redis` and `docker volume rm rag-redis-data`. Your documents are not in Redis: they are in `data/` (the vector database in `data/lancedb`, the PDFs in `data/pdfs`).

## How search works

A question is answered from the 8 best chunks of the tenant's documents. They come from three rankings that are merged (reciprocal rank fusion): the closest chunks by meaning (the embedding vectors), and the best keyword matches in an Italian column and in an English column.

- **Why both kinds.** Vectors find the same meaning in different words, and even in a different language, which is how an English question finds Italian text. Keywords find exact strings such as case numbers, names and codes, which vectors are weak at.
- **Why two keyword columns.** Each chunk's language is detected when it is ingested (Italian or English, with the `lingua` library, offline), and the language most chunks of a PDF were detected in is given to all of its chunks, because a document is normally written in one language. Each chunk goes into the column of its document's language, and each column has an index that stems words for that language, so *rifiuto* finds *rifiuti* and *device* finds *devices*. A PDF where the detector is unsure about every chunk goes into both columns. A quoted phrase in a question, such as `"copia privata"`, is matched as an exact phrase.
- **Weights.** Each keyword ranking counts for half of the vector ranking, so together they weigh the same. Each ranking is 8 results deep. On ten test questions over your Italian documents, deeper rankings or full-weight keywords made the results worse. That is a small sample, so the settings are only reasonable defaults.
- **Known limit.** A PDF that really mixes both languages is searched only with the rules of its main language.

Tables made before the language columns existed are upgraded by `uv run rag reindex --tenant <uuid>`. It recomputes every chunk's language and rebuilds both indexes, and it is safe to run again. Stop the worker first, because LanceDB allows one writer. Adding a document to an old table upgrades it too, but searching one fails with a message that asks for the reindex.

## Memory requirements

Everything except the OpenAI call runs on your machine, and the models are loaded inside the Python process. That makes RAM the main resource to plan for.

Measured on a 16-core Linux machine with 32 GB of RAM, CPU only, using one 33-page PDF:

| Activity | Peak RAM (RSS) | Notes |
|---|---|---|
| Ingesting a PDF | about 4.0 GB | Docling layout models plus the embedding model. The 33-page PDF took about 40 s. |
| Answering a question | about 1.6 GB | Embedding model and LanceDB only. The LLM runs at OpenAI, not locally. |
| Streamlit app, idle after first use | about 1.9 GB | Models stay loaded between questions. |
| Phoenix server | about 0.5 GB | Grows slowly as traces accumulate. |

Recommendations:

- **8 GB of RAM** is a sensible minimum to run Streamlit, the ingestion worker and Phoenix together on one machine. Ingestion runs in the worker process (see below), so the 4 GB peak belongs to the worker and the Streamlit process stays at its idle size. The Streamlit figure above was measured before ingestion moved to the worker, so it is probably a little lower now.
- **16 GB** leaves comfortable room for larger PDFs and other applications.
- If you do not need traces, set `RAG_TRACING=0` and do not run Phoenix.

Not measured: PDFs much longer than 33 pages. Parsing memory is likely to grow with page count, so test with your largest document before sizing a server.

Disk space, measured:

- The Python environment is about 7 GB, mostly PyTorch.
- Downloaded model files are about 1.4 GB in `~/.cache/huggingface`. That folder may also contain models from other projects, so the real figure for this app can be smaller.
- The vector database is small: 152 chunks from 5 PDFs use about 1.3 MB.
- The first ingest downloads the models, so it is slower than later ones.

## Ingestion queue

A PDF is submitted as a small message and the call returns at once. A separate worker process downloads it, then does the parsing and embedding. The sidebar of the app lists each job as queued, running, retrying, done, failed or cancelled.

The queue is Celery with Redis as the broker and the result backend (`RAG_REDIS_URL` in `.env`). Start Redis, then the worker next to the app: `uv run rag worker`. The worker takes one job at a time, in a single process. LanceDB allows a single writer, and ingesting a PDF peaks at about 4 GB of RAM, so do not run more than one worker.

### Submitting

`scripts/produce.py` is a throwaway producer that puts URLs on the queue:

```
uv run python scripts/produce.py --tenant <company uuid> https://files.example.com/contract.pdf
```

The message is the task `rag.ingest` with two JSON arguments, `tenant` and `url`. The PDF never travels in the message: the worker downloads it, ingests it, and keeps it in the tenant's folder (the app needs it to show the cited page). The PDF is kept only if ingestion worked. A job that fails leaves no file behind, and a failed job never overwrites the PDF of a document that was ingested earlier. Submitting a URL with the same file name again replaces that document once the new version has been ingested. Jobs queued while the worker is stopped wait in Redis and run when it starts.

Job states are kept in a small SQLite file (`data/jobs.sqlite`), because Celery's result backend cannot list jobs by tenant. A producer on another machine creates no row, so such a job appears in the sidebar once the worker starts it.

### What the worker accepts

The URL comes from a producer, so the worker treats it as hostile (this is what stops it being used to reach your internal network):

- Only `https` on the default port, with no credentials in the URL.
- The host name is resolved by the worker, and only public addresses are contacted. Private, loopback and link-local ranges (including the cloud metadata address) are refused, and so are redirects that lead there. Up to 3 redirects are followed.
- The file must start with `%PDF-` and stay below `RAG_MAX_PDF_MB` (50 by default). The download has a time limit.
- Exception: to fetch from a server on your own network (a MinIO, a NAS), list its host name in `RAG_FETCH_ALLOW`, comma separated, for example `RAG_FETCH_ALLOW=minio.internal,files.lan`. Only those exact host names skip the public-address check, and they may also use `http` and any port. A redirect to any other host is still checked as usual.

A job that can never succeed (a blocked address, a 4xx answer, not a PDF, no extractable text, a malformed message) is marked failed at once and not retried. A temporary problem (timeout, 5xx, connection error) and any other error is retried with a growing delay, up to `RAG_JOB_ATTEMPTS` attempts in total. A job that fails permanently still shows as a successful Celery task, because the worker handled it: the real outcome is in the job status.

### Running Redis safely

- Messages use JSON only, never pickle, so a message cannot run code in the worker.
- The queue is database 0 and the job results are database 1 of the same server (`RAG_REDIS_RESULTS_URL` changes that). Databases separate the data, so clearing one leaves the other, but they are not a security boundary: a password and the `noeviction` setting apply to the whole server. Redis Cluster has only database 0, so set `RAG_REDIS_RESULTS_URL` there.
- Redis is the trust boundary. Anyone who can write to it can queue a job for any tenant, because the tenant is part of the message. Keep Redis on a private network or `127.0.0.1`, set a password, and give a producer an ACL user that may only add to the queue.
- Turn on persistence (`appendonly yes`), or a Redis restart empties the queue. Set `maxmemory-policy noeviction` so Redis never drops queued jobs to free memory.
- A job that is not acknowledged within `RAG_VISIBILITY_SECONDS` (one hour by default) is handed out again, so keep it above your slowest PDF. Reprocessing is harmless, because ingesting a PDF replaces its chunks.

## To do

1. **Keep the PDFs in external storage.** Today the PDFs are written to the container's disk (`data/pdfs`), and the app needs them to show the cited page. Store them in an object storage with an S3 interface (MinIO, Garage, Cloudflare R2, AWS S3 and so on), so the container's filesystem does not grow with every document.
2. **Support more than one cloud LLM provider.** The answer step is tied to OpenAI. Make the provider configurable (for example OpenAI, Anthropic, Google), chosen in `.env`.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

import rag.ask as ask_mod
from rag.ingest import ingest
from rag.search import search

EN = "The warranty period for the X200 pump is 36 months from the delivery date."
IT = "Il periodo di garanzia della pompa X200 e di 36 mesi dalla data di consegna."
STYLES = getSampleStyleSheet()
BODY, H1 = STYLES["BodyText"], STYLES["Heading1"]
T1 = "11111111-1111-4111-8111-111111111111"
T2 = "22222222-2222-4222-8222-222222222222"
FILLER = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 6


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag")
    path = tmp / "manual.pdf"
    story = [Paragraph("Chapter 1", H1), Paragraph(FILLER, BODY)]
    story += [PageBreak(), Paragraph("Warranty", H1), Paragraph(EN, BODY), Paragraph(IT, BODY)]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.lancedb_store.DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr("rag.store.PDF_DIR", str(tmp_path / "pdfs"))
    monkeypatch.setattr("rag.jobs.JOBS_DB", str(tmp_path / "jobs.sqlite"))


def test_search_returns_paragraph_with_page_and_ref(pdf):
    assert ingest(pdf, T1) > 0
    top = search("how long is the warranty of the X200?", T1, k=3)[0]
    assert "36 months" in top["text"]
    assert top["page"] == 2
    assert json.loads(top["items"])[0]["ref"].startswith("#/texts/")


def test_ingest_twice_does_not_duplicate(pdf):
    n = ingest(pdf, T1)
    ingest(pdf, T1)
    assert len(search("warranty", T1, k=50)) == n


def test_ask_flags_invented_citations(pdf, monkeypatch):
    ingest(pdf, T1)
    fake = ask_mod.Answer(
        question_language="English",
        answer="36 months [1] and more [99]",
        citations=[
            ask_mod.Citation(chunk=1, quote="warranty period for the X200 pump"),
            ask_mod.Citation(chunk=1, quote="this sentence is not in the source"),
            ask_mod.Citation(chunk=99, quote="never retrieved"),
        ],
    )
    monkeypatch.setattr(
        ask_mod,
        "OpenAI",
        lambda: SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: SimpleNamespace(output_parsed=fake))),
    )
    r = ask_mod.ask("warranty X200", T1, k=1)
    assert r["answer"] == "36 months [1] and more "  # unknown marker [99] removed
    assert len(r["sources"]) == 1  # two citations of chunk 1 are one reference, chunk 99 is dropped
    assert [q["verified"] for q in r["sources"][0]["quotes"]] == [True, False]



def test_highlight_renders_png(pdf):
    from rag.highlight import render_page

    ingest(pdf, T1)
    items = json.loads(search("warranty X200", T1, k=1)[0]["items"])
    assert render_page(str(pdf), items[0]["page"], items)[:4] == b"\x89PNG"


def test_streamlit_chat_shows_answer_and_source(pdf, monkeypatch):
    from streamlit.testing.v1 import AppTest

    ingest(pdf, T1)
    monkeypatch.setenv("RAG_TRACING", "0")
    hit = search("warranty X200", T1, k=2)
    fake = {
        "answer": "36 months [1]",
        "sources": [{"n": 1, "doc": hit[0]["doc"], "page": hit[0]["page"], "heading": "Warranty", "path": hit[0]["path"],
                     "items": json.loads(hit[0]["items"]), "quotes": [{"text": "36 months", "verified": True}]}],
        "chunks": hit, "seconds": 0.1, "trace_id": "abc",
    }
    monkeypatch.setattr(ask_mod, "ask", lambda q, t, k=8: fake)
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60)
    at.query_params["tenant"] = T1  # how the frontends will embed the app
    at.run()
    assert not at.sidebar.selectbox  # embedded: no dropdown, so other companies are never listed
    at.chat_input[0].set_value("warranty?").run()
    assert not at.exception
    assert "36 months [1]" in [m.value for m in at.markdown]
    assert any("p.2" in e.label for e in at.expander)


def test_ingested_docs_counts_pdfs_across_sessions(pdf):
    from rag.store import ingested_docs

    assert not ingested_docs(T1)  # empty db, no table yet
    n = ingest(pdf, T1)
    assert ingested_docs(T1) == {pdf.name: n}


def test_references_are_renumbered_without_holes(monkeypatch):
    chunks = [{"id": f"d:{i}", "doc": f"doc{i}.pdf", "path": "x", "page": i, "heading": "", "text": f"text {i}",
               "items": "[]", "score": 0.1} for i in range(1, 6)]
    monkeypatch.setattr(ask_mod, "search", lambda q, t, k=8: chunks)
    fake = ask_mod.Answer(
        question_language="English",
        answer="A [3] B [2][5] C [3, 5]",  # the model cited excerpts 3 and 5; [2] has no citation
        citations=[ask_mod.Citation(chunk=5, quote="text 5"), ask_mod.Citation(chunk=3, quote="text 3")],
    )
    monkeypatch.setattr(
        ask_mod,
        "OpenAI",
        lambda: SimpleNamespace(responses=SimpleNamespace(parse=lambda **_: SimpleNamespace(output_parsed=fake))),
    )
    r = ask_mod.ask("q", T1)
    assert r["answer"] == "A [1] B [2] C [1][2]"
    assert [(s["n"], s["doc"]) for s in r["sources"]] == [(1, "doc3.pdf"), (2, "doc5.pdf")]


def test_reasoning_effort_comes_from_env_setting(monkeypatch):
    monkeypatch.setattr(ask_mod, "REASONING_EFFORT", "high")
    monkeypatch.setattr(ask_mod, "search", lambda q, t, k=8: [])
    seen = {}

    def parse(**kw):
        seen.update(kw)
        return SimpleNamespace(output_parsed=ask_mod.Answer(question_language="English", answer="x", citations=[]))

    monkeypatch.setattr(ask_mod, "OpenAI", lambda: SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    ask_mod.ask("q", T1)
    assert seen["reasoning"] == {"effort": "high"}


def test_each_tenant_only_sees_its_own_documents(pdf):
    from rag.store import ingested_docs

    ingest(pdf, T1)
    assert ingested_docs(T1) and not ingested_docs(T2)
    assert not ingested_docs(T2)
    with pytest.raises(Exception):
        search("warranty", T2)  # no table for this tenant, never falls back to another one's data
    ingest(pdf, T2)
    assert ingested_docs(T1) == ingested_docs(T2) == {pdf.name: len(search("warranty", T1, k=50))}


@pytest.mark.parametrize("bad", ["../x", "", "not-a-uuid", "../../etc", "11111111-1111-4111-8111-11111111111/"])
def test_tenant_must_be_a_uuid(bad):
    from rag.store import ingested_docs, tenant_id

    with pytest.raises(ValueError):
        tenant_id(bad)
    with pytest.raises(ValueError):
        ingested_docs(bad)


def test_dropdown_shows_label_but_value_is_the_uuid(pdf, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    f = tmp_path / "tenants.json"
    f.write_text(json.dumps({T1: "acme", T2.upper(): "other"}))
    monkeypatch.setattr("rag.store.TENANTS_FILE", str(f))
    monkeypatch.setenv("RAG_TRACING", "0")
    ingest(pdf, T1)
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60).run()
    box = at.sidebar.selectbox[0]
    assert box.options == ["acme", "other"] and box.value == T1  # T2 is listed although it has no documents yet
    assert not at.sidebar.button and not at.sidebar.text_input  # no "Add tenant" widget


def test_invalid_tenant_in_url_is_refused(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RAG_TRACING", "0")
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60)
    at.query_params["tenant"] = "../../etc"
    at.run()
    assert at.error and not at.chat_input


class FakeStore:
    """A throwaway in-memory backend: if the pipeline only talks to the VectorStore interface, this is enough."""

    def __init__(self):
        self.rows = {}  # tenant -> list of rows

    def replace_doc(self, tenant, doc, rows):
        self.rows[tenant] = [r for r in self.rows.get(tenant, []) if r["doc"] != doc] + rows

    def search(self, tenant, text, vector, k):
        if tenant not in self.rows:
            raise LookupError(tenant)
        hits = sorted(self.rows[tenant], key=lambda r: -sum(w in r["text"].lower() for w in text.lower().split()))
        return [{key: r[key] for key in ("id", "doc", "path", "page", "heading", "text", "items")} | {"score": 1.0}
                for r in hits[:k]]

    def docs(self, tenant):
        from collections import Counter

        return Counter(r["doc"] for r in self.rows.get(tenant, []))

    def tenants(self):
        return sorted(self.rows)

    def reindex(self, tenant):
        from collections import Counter

        return Counter()


def test_pipeline_works_with_any_backend_that_implements_the_interface(pdf, monkeypatch, tmp_path):
    fake = FakeStore()
    monkeypatch.setattr("rag.store.backend", lambda: fake)
    n = ingest(pdf, T1)
    top = search("warranty X200 months", T1, k=1)[0]
    assert "36 months" in top["text"] and top["page"] == 2
    from rag.store import ingested_docs, tenants

    assert ingested_docs(T1) == {pdf.name: n} and not ingested_docs(T2)
    assert T1 in tenants()
    assert not (tmp_path / "db").exists()  # LanceDB was never touched


def test_unknown_backend_name_is_refused(monkeypatch):
    from rag import store

    store.backend.cache_clear()
    monkeypatch.setattr(store, "BACKEND", "nope")
    with pytest.raises(ValueError):
        store.backend()
    store.backend.cache_clear()


# ---- jobs: the worker and the Celery task ----

def add_job(job_id, tenant, state="queued"):
    from contextlib import closing

    from rag import jobs

    with closing(jobs._conn()) as c, c:
        c.execute("insert into jobs (id, tenant, doc, state, created, updated) values (?, ?, 'x.pdf', ?, '', '')",
                  (job_id, tenant, state))


def state_of(tenant):
    from rag import jobs

    return [(j["state"], j["detail"]) for j in jobs.jobs_for(tenant)]


@pytest.fixture
def site(pdf, monkeypatch):
    """A local web server for the worker to download from. Loopback is normally refused, so this relaxes the guard."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    content = pdf.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            routes = {
                "/doc.pdf": (200, content), "/Q1 report.pdf".replace(" ", "%20"): (200, content),
                "/text.pdf": (200, b"hello, not a pdf"), "/missing.pdf": (404, b""), "/boom.pdf": (503, b""),
            }
            if path in ("/go", "/evil", "/loop"):
                self.send_response(302)
                self.send_header("Location", {"/go": "/doc.pdf", "/evil": "http://169.254.169.254/latest", "/loop": "/loop"}[path])
                self.end_headers()
                return
            status, body = routes.get(path, (404, b""))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("rag.fetch.ALLOW", {"127.0.0.1"})
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()



@pytest.fixture
def eager():
    """Celery runs a task in the calling process, so the whole path from submit_url to a searchable PDF is testable
    without a Redis server. Reset by hand: pytest's monkeypatch cannot restore Celery's settings object."""
    from rag.celery_app import app

    app.conf.task_always_eager = True
    yield
    app.conf.task_always_eager = False


def test_worker_downloads_and_ingests_a_url_job(site):
    from rag import worker
    from rag.store import ingested_docs

    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})
    assert state_of(T1)[0][0] == "done" and state_of(T1)[0][1].endswith("chunks") and ingested_docs(T1)


def test_worker_creates_the_status_row_for_a_message_from_another_machine(site):
    from rag import worker

    worker.process({"job_id": "from-elsewhere", "tenant": T1, "url": f"{site}/doc.pdf"})  # no row exists yet
    assert state_of(T1)[0][0] == "done"


@pytest.mark.parametrize("job", [
    {"tenant": "nope", "url": "https://x/a.pdf"},
    {"url": "https://x/a.pdf"},
    {"tenant": T1},
    {"tenant": T1, "url": 5},  # JSON can carry any type
    {"tenant": T1, "path": "/etc/passwd"},  # a local path is not a job any more
])
def test_worker_marks_a_bad_job_failed_without_ingesting(job):
    from rag import worker
    from rag.store import ingested_docs

    add_job("j1", T1)
    worker.process({"job_id": "j1", **job})  # returns normally: a bad message must not be retried
    assert state_of(T1)[0] == ("failed", "invalid job")
    assert not ingested_docs(T1)


def test_worker_marks_a_blocked_url_failed_without_retry():
    from rag import worker

    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": "https://127.0.0.1/a.pdf"}, final=False)  # does not raise
    assert state_of(T1)[0] == ("failed", "127.0.0.1 resolves to a non-public address")


def test_worker_retries_a_temporary_failure_and_only_the_last_attempt_is_failed(site):
    from rag import worker

    add_job("j1", T1)
    job = {"job_id": "j1", "tenant": T1, "url": f"{site}/boom.pdf"}  # the server answers 503
    with pytest.raises(Exception):
        worker.process(job, final=False)
    assert state_of(T1)[0][0] == "retrying"
    with pytest.raises(Exception):
        worker.process(job, final=True)
    assert state_of(T1)[0][0] == "failed"


def test_pdf_without_text_is_failed_and_not_retried(site, monkeypatch):
    from rag import worker
    from rag.ingest import NoTextError

    def no_text(*a):
        raise NoTextError("no text extracted")

    monkeypatch.setattr("rag.ingest.ingest", no_text)
    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"}, final=False)  # does not raise
    assert state_of(T1)[0] == ("failed", "no text extracted")


def pdf_files(tenant):
    from rag.store import pdf_dir

    folder = pdf_dir(tenant)
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_a_successful_job_keeps_the_pdf_under_its_real_name(site, pdf):
    from rag import worker
    from rag.search import search
    from rag.store import pdf_dir

    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})
    assert pdf_files(T1) == ["doc.pdf"]  # the hidden staging file is gone
    assert (pdf_dir(T1) / "doc.pdf").read_bytes() == pdf.read_bytes()
    hit = search("warranty X200", T1, k=1)[0]
    assert hit["doc"] == "doc.pdf" and hit["path"] == str((pdf_dir(T1) / "doc.pdf").resolve())  # the app finds the PDF


def test_a_job_that_fails_leaves_no_pdf_behind(site, monkeypatch):
    from rag import worker
    from rag.ingest import NoTextError

    def no_text(*a):
        raise NoTextError("no text extracted")

    monkeypatch.setattr("rag.ingest.ingest", no_text)
    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})  # downloaded fine, nothing to ingest
    assert state_of(T1)[0][0] == "failed" and pdf_files(T1) == []


def test_a_retried_job_leaves_no_pdf_behind_either(site, monkeypatch):
    from rag import worker

    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr("rag.ingest.ingest", boom)
    add_job("j1", T1)
    with pytest.raises(RuntimeError):
        worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"}, final=False)
    assert state_of(T1)[0][0] == "retrying" and pdf_files(T1) == []  # the retry downloads it again


def test_a_failed_download_leaves_nothing_behind(site):
    from rag import worker

    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/text.pdf"})  # the URL serves text, not a PDF
    assert state_of(T1)[0][0] == "failed" and pdf_files(T1) == []


def test_a_failed_job_never_destroys_the_pdf_of_an_earlier_good_ingestion(site, pdf, monkeypatch):
    from rag import worker
    from rag.ingest import NoTextError
    from rag.store import ingested_docs, pdf_dir

    add_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})
    chunks = ingested_docs(T1)["doc.pdf"]

    def no_text(*a):
        raise NoTextError("no text extracted")

    monkeypatch.setattr("rag.ingest.ingest", no_text)  # the same URL now serves something that cannot be ingested
    add_job("j2", T1)
    worker.process({"job_id": "j2", "tenant": T1, "url": f"{site}/doc.pdf"})
    assert sorted(s for s, _ in state_of(T1)) == ["done", "failed"]
    assert (pdf_dir(T1) / "doc.pdf").read_bytes() == pdf.read_bytes()  # the good PDF is still there
    assert ingested_docs(T1)["doc.pdf"] == chunks and pdf_files(T1) == ["doc.pdf"]


def test_submit_url_sends_a_json_message_with_the_url_unchanged(monkeypatch):
    from rag import jobs
    from rag.celery_app import ingest_pdf

    sent = []
    monkeypatch.setattr(ingest_pdf, "apply_async", lambda **kw: sent.append(kw))
    url = "https://files.example.com/a/Report%20Q1.pdf?X-Amz-Signature=abc&x=1"
    job_id = jobs.submit_url(T1, url)
    assert sent == [{"kwargs": {"tenant": T1, "url": url}, "task_id": job_id}]
    assert state_of(T1) == [("queued", "")] and jobs.jobs_for(T1)[0]["doc"] == "Report Q1.pdf"
    assert not jobs.jobs_for(T2)


@pytest.mark.parametrize("url", ["ftp://example.com/a.pdf", "not a url", "/etc/passwd", ""])
def test_submit_url_refuses_something_that_is_not_a_web_address(url, monkeypatch):
    from rag import jobs
    from rag.celery_app import ingest_pdf

    monkeypatch.setattr(ingest_pdf, "apply_async", lambda **kw: pytest.fail("must not be queued"))
    with pytest.raises(ValueError):
        jobs.submit_url(T1, url)
    assert not jobs.jobs_for(T1)


def test_submit_url_marks_the_job_failed_when_redis_is_unreachable(monkeypatch):
    from rag import jobs
    from rag.celery_app import ingest_pdf

    def down(**kw):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(ingest_pdf, "apply_async", down)
    with pytest.raises(ConnectionError):
        jobs.submit_url(T1, "https://example.com/a.pdf")
    assert state_of(T1)[0][0] == "failed" and "redis is down" in state_of(T1)[0][1]


def test_a_queued_job_goes_through_celery_to_a_searchable_pdf(site, eager):
    from rag import jobs
    from rag.store import ingested_docs

    job_id = jobs.submit_url(T1, f"{site}/doc.pdf")
    assert state_of(T1)[0][0] == "done" and jobs.jobs_for(T1)[0]["id"] == job_id  # the Celery task id is the job id
    assert ingested_docs(T1)


def test_celery_retries_a_temporary_failure_then_gives_up(eager, monkeypatch):
    from rag import worker
    from rag.celery_app import ATTEMPTS, ingest_pdf
    from rag.fetch import Transient

    calls = []

    def down(*a):
        calls.append(1)
        raise Transient("host unreachable")

    monkeypatch.setattr(worker, "fetch_pdf", down)
    add_job("j1", T1)
    result = ingest_pdf.apply(kwargs={"tenant": T1, "url": "https://example.com/a.pdf"}, task_id="j1")
    assert result.failed() and len(calls) == ATTEMPTS  # three attempts in all, by default
    assert state_of(T1)[0][0] == "failed" and "host unreachable" in state_of(T1)[0][1]


def test_celery_only_speaks_json():
    from rag.celery_app import app

    assert app.conf.accept_content == ["json"] and app.conf.task_serializer == "json"  # a pickle message is refused
    assert app.conf.worker_prefetch_multiplier == 1 and app.conf.task_acks_late


def test_throwaway_producer_queues_every_url(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    from rag.celery_app import ingest_pdf

    spec = importlib.util.spec_from_file_location("produce", Path(__file__).parent.parent / "scripts" / "produce.py")
    produce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(produce)
    sent = []
    monkeypatch.setattr(ingest_pdf, "apply_async", lambda **kw: sent.append(kw["kwargs"]))
    produce.main(["--tenant", T1.upper(), "https://h/a.pdf", "https://h/b.pdf"])
    assert sent == [{"tenant": T1, "url": "https://h/a.pdf"}, {"tenant": T1, "url": "https://h/b.pdf"}]
    assert capsys.readouterr().out.count(" -> job ") == 2


# ---- the worker downloads, with SSRF guards ----

@pytest.mark.parametrize("url", [
    "http://example.com/a.pdf",  # not https
    "ftp://example.com/a.pdf",
    "https://127.0.0.1/a.pdf",
    "https://localhost/a.pdf",
    "https://10.0.0.5/a.pdf",
    "https://192.168.1.10/a.pdf",
    "https://169.254.169.254/latest/meta-data/",  # the cloud metadata address
    "https://[::1]/a.pdf",
    "https://[::ffff:10.0.0.1]/a.pdf",  # a private IPv4 address written as IPv6
    "https://user:secret@example.com/a.pdf",
    "https://example.com:8080/a.pdf",  # not the default https port
])
def test_fetch_refuses_urls_that_point_inside(url, tmp_path):
    from rag.fetch import Refused, fetch_pdf

    with pytest.raises(Refused):
        fetch_pdf(url, tmp_path / "t" / "a.pdf")
    assert not list((tmp_path / "t").glob("*"))


def test_fetch_downloads_a_pdf_to_the_file_it_is_given(site, tmp_path, pdf):
    from rag.fetch import fetch_pdf

    dest = tmp_path / "t" / "a.pdf"
    fetch_pdf(f"{site}/Q1%20report.pdf?sig=abc", dest)
    assert dest.read_bytes() == pdf.read_bytes()
    assert [p.name for p in dest.parent.iterdir()] == ["a.pdf"]  # no partial file left


def test_fetch_follows_a_redirect_but_not_into_a_blocked_address(site, tmp_path, monkeypatch):
    from rag import fetch
    from rag.fetch import Refused, fetch_pdf

    fetch_pdf(f"{site}/go", tmp_path / "t" / "a.pdf")
    assert (tmp_path / "t" / "a.pdf").read_bytes()[:5] == b"%PDF-"
    with pytest.raises(Refused):
        fetch_pdf(f"{site}/evil", tmp_path / "t2" / "a.pdf")  # redirects to the cloud metadata address
    with pytest.raises(Refused):
        fetch_pdf(f"{site}/loop", tmp_path / "t3" / "a.pdf")  # redirects forever


@pytest.mark.parametrize("path,error", [("/text.pdf", "Refused"), ("/missing.pdf", "Refused"), ("/boom.pdf", "Transient")])
def test_fetch_tells_permanent_from_temporary_failures(site, tmp_path, path, error):
    from rag import fetch

    with pytest.raises(getattr(fetch, error)):
        fetch.fetch_pdf(site + path, tmp_path / "t" / "a.pdf")
    assert not list((tmp_path / "t").glob("*"))


def test_fetch_stops_at_the_size_limit(site, tmp_path, monkeypatch):
    from rag import fetch

    monkeypatch.setattr(fetch, "MAX_PDF_BYTES", 100)
    with pytest.raises(fetch.Refused):
        fetch.fetch_pdf(f"{site}/doc.pdf", tmp_path / "t" / "a.pdf")
    assert not list((tmp_path / "t").glob("*"))


def test_document_name_from_a_url():
    from rag.fetch import doc_name

    assert doc_name("https://h/x/Report.PDF?a=b") == "Report.PDF"
    assert doc_name("https://h/x/..%2f..%2fevil.pdf") == "evil.pdf"  # only the last path segment counts
    assert doc_name("https://h/download?id=1") == doc_name("https://h/download?id=1") != doc_name("https://h/download?id=2")
    assert doc_name("https://h/download?id=1").endswith(".pdf")


def test_sidebar_shows_ingestion_jobs(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RAG_TRACING", "0")
    add_job("j1", T1, state="queued")
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60)
    at.query_params["tenant"] = T1
    at.run()
    assert not at.exception
    assert any(c.value == "queued" for c in at.sidebar.caption)  # one line per job: its name, then its state
    assert any("uv run rag worker" in c.value for c in at.sidebar.caption)




@pytest.mark.parametrize("broker,explicit,expected", [
    ("redis://localhost:6379/0", None, "redis://localhost:6379/1"),
    ("redis://:secret@redis.internal:6380/0", None, "redis://:secret@redis.internal:6380/1"),
    ("redis://localhost:6379", None, "redis://localhost:6379/1"),  # no database in the URL
    ("rediss://host:6379/2", None, "rediss://host:6379/1"),
    ("redis://localhost:6379/0", "redis://other:6379/5", "redis://other:6379/5"),  # an explicit URL wins
])
def test_results_go_to_redis_database_1_unless_told_otherwise(broker, explicit, expected):
    from rag.celery_app import results_url

    assert results_url(broker, explicit) == expected


def test_the_queue_and_the_results_use_different_redis_databases():
    from rag.celery_app import REDIS_URL, app

    assert app.conf.broker_url == REDIS_URL and app.conf.result_backend.rstrip("/").endswith("/1")


def test_the_answer_language_follows_the_question_not_the_excerpts():
    from rag.ask import INSTRUCTIONS, Answer

    assert list(Answer.model_fields)[0] == "question_language"  # decided before the answer is written
    assert "language of the QUESTION" in INSTRUCTIONS and "not the language of the excerpts" in INSTRUCTIONS
    assert "verbatim" in INSTRUCTIONS  # quotes keep the excerpt's language, so they can still be verified


def test_dropdown_opens_on_the_default_company(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    f = tmp_path / "tenants.json"
    f.write_text(json.dumps({T1: "contoso", T2: "acme"}))  # T1 sorts first, but acme is the default
    monkeypatch.setattr("rag.store.TENANTS_FILE", str(f))
    monkeypatch.setenv("RAG_TRACING", "0")
    monkeypatch.setattr("rag.store.DEFAULT_TENANT", T2.upper())  # any spelling of the UUID works
    box = AppTest.from_file("../src/rag/app.py", default_timeout=60).run().sidebar.selectbox[0]
    assert box.options == ["contoso", "acme"] and box.value == T2
    monkeypatch.setattr("rag.store.DEFAULT_TENANT", "33333333-3333-4333-8333-333333333333")  # not a known company
    assert AppTest.from_file("../src/rag/app.py", default_timeout=60).run().sidebar.selectbox[0].value == T1
    monkeypatch.setattr("rag.store.DEFAULT_TENANT", None)
    assert AppTest.from_file("../src/rag/app.py", default_timeout=60).run().sidebar.selectbox[0].value == T1


def test_a_malformed_default_company_is_an_error_not_ignored(monkeypatch):
    from rag import store

    monkeypatch.setattr(store, "DEFAULT_TENANT", "acme")  # the label, not the UUID: a typo worth refusing
    with pytest.raises(ValueError):
        store.default_tenant()


# ---- language columns and the merged keyword + vector search ----

IT_TEXT = "I rifiuti prodotti dal trattamento meccanico sono stati classificati correttamente dal giudice nazionale."
EN_TEXT = "The court held that the devices were charged unlawfully, and the courts agreed with the manufacturer."


def lance_rows(*texts):
    return [{"id": f"d:{i}", "doc": "d.pdf", "path": "/x/d.pdf", "page": 1, "heading": "", "text": t, "items": "[]",
             "vector": [0.0] * 384} for i, t in enumerate(texts)]


@pytest.mark.parametrize("text,expected", [
    (IT_TEXT, "it"), (EN_TEXT, "en"),
    ("1", None), ("C-110/15", None), ("Pubblicato il 01/07/2020", None),  # too little text to judge
])
def test_language_detection_says_unsure_for_short_text(text, expected):
    from rag.language import detect

    assert detect(text) == expected


def test_every_chunk_takes_the_language_of_its_document():
    from rag.lancedb_store import _with_language_columns

    rows = _with_language_columns(lance_rows(IT_TEXT, IT_TEXT + " Ancora.", EN_TEXT, "1", "C-110/15 Smith v Jones"))
    assert {r["lang"] for r in rows} == {"it"}  # the English chunk and the short ones follow the majority
    assert all(r["text_it"] == r["text"] and r["text_en"] == "" for r in rows)


def test_two_documents_are_judged_separately():
    from rag.lancedb_store import _with_language_columns

    other = [{**r, "doc": "e.pdf"} for r in lance_rows(EN_TEXT, "1")]
    rows = _with_language_columns(lance_rows(IT_TEXT) + other)
    assert [r["lang"] for r in rows] == ["it", "en", "en"]


def test_a_document_with_no_confident_chunk_is_searchable_in_both_columns():
    from rag.lancedb_store import _with_language_columns

    (r,) = _with_language_columns(lance_rows("1"))
    assert (r["lang"], r["text_it"], r["text_en"]) == ("", "1", "1")


def test_keyword_search_stems_in_each_language():
    from rag.lancedb_store import LanceStore

    store = LanceStore()
    store.replace_doc(T1, "d.pdf", lance_rows(IT_TEXT, "Un altro paragrafo senza alcuna parola in comune."))
    store.replace_doc(T1, "e.pdf", [{**r, "doc": "e.pdf", "id": "e:0"} for r in lance_rows(EN_TEXT)])
    top = lambda q: store.search(T1, q, [0.0] * 384, 1)[0]["text"]
    assert top("rifiuto") == IT_TEXT  # singular finds 'rifiuti': the Italian stemmer
    assert top("device") == EN_TEXT  # singular finds 'devices': the English stemmer
    assert top("courts") == EN_TEXT and top("giudici") == IT_TEXT


def test_a_vector_match_and_a_keyword_match_both_count():
    from rag.lancedb_store import LanceStore

    store = LanceStore()
    rows = lance_rows("Testo senza rapporto con la domanda posta qui sopra.", "Altro testo diverso per niente simile.",
                      "Una frase che contiene la parola rarissima quokka e basta.")
    rows[0]["vector"] = [1.0] + [0.0] * 383  # nearest to the question by meaning
    store.replace_doc(T1, "d.pdf", rows)
    texts = [r["text"] for r in store.search(T1, "quokka", [1.0] + [0.0] * 383, 3)]
    assert set(texts[:2]) == {rows[0]["text"], rows[2]["text"]}  # the vector hit and the keyword hit lead the ranking


@pytest.mark.parametrize("q", ['"315/20"', 'what is "this" (and *that*)?', "a OR b", ":", "'", "()", "***", "?", "-", "\\", "e" * 500])
def test_odd_question_text_does_not_break_the_keyword_search(q):
    from rag.lancedb_store import LanceStore

    store = LanceStore()
    store.replace_doc(T1, "d.pdf", lance_rows(IT_TEXT, EN_TEXT))
    assert isinstance(store.search(T1, q, [0.0] * 384, 3), list)


def test_an_old_table_without_language_columns_is_upgraded_not_lost():
    import lancedb

    from rag import lancedb_store
    from rag.lancedb_store import LanceStore
    from rag.store import ingested_docs

    folder = Path(lancedb_store.DB_DIR) / T1
    folder.mkdir(parents=True)
    old = lancedb.connect(str(folder)).create_table("chunks", lance_rows(IT_TEXT))  # the layout before this change
    assert "text_it" not in old.schema.names
    store = LanceStore()
    with pytest.raises(RuntimeError, match="rag reindex"):
        store.search(T1, "rifiuti", [0.0] * 384, 3)  # it says what to do, instead of failing obscurely
    new = lance_rows(EN_TEXT)
    new[0]["id"], new[0]["doc"] = "e:0", "e.pdf"
    store.replace_doc(T1, "e.pdf", new)  # adding a document upgrades the table first
    assert ingested_docs(T1) == {"d.pdf": 1, "e.pdf": 1}  # the old row is still there
    assert [r["text"] for r in store.search(T1, "rifiuto", [0.0] * 384, 1)] == [IT_TEXT]


def test_reindex_reports_the_languages_and_works_twice():
    from rag.lancedb_store import LanceStore

    store = LanceStore()
    assert store.reindex(T1) == {}  # no table yet
    store.replace_doc(T1, "d.pdf", lance_rows(IT_TEXT, IT_TEXT + " Ancora.", "1"))
    store.replace_doc(T1, "e.pdf", [{**r, "doc": "e.pdf", "id": "e:0"} for r in lance_rows(EN_TEXT)])
    assert store.reindex(T1) == {"it": 3, "en": 1} and store.reindex(T1) == {"it": 3, "en": 1}
    assert [r["text"] for r in store.search(T1, "device", [0.0] * 384, 1)] == [EN_TEXT]


def test_a_quoted_phrase_must_match_exactly():
    from rag.lancedb_store import LanceStore

    store = LanceStore()
    store.replace_doc(T1, "d.pdf", lance_rows("The levy on private copying of devices is unlawful here.",
                                              "Copying is private, and a levy on devices is unlawful there."))
    hits = [r["text"] for r in store.search(T1, '"private copying"', [0.0] * 384, 2)]
    assert hits[0].startswith("The levy on private copying")  # the phrase, in order, ranks first


# ---- removing and cancelling jobs from the list ----

def test_only_a_finished_job_can_be_removed_from_the_list():
    from rag import jobs

    for job_id, state in [("a", "done"), ("b", "failed"), ("c", "cancelled"), ("d", "queued"), ("e", "running"),
                          ("f", "retrying"), ("g", "cancelling")]:
        add_job(job_id, T1, state)
    removed = {j: jobs.delete_job(j, T1) for j in "abcdefg"}
    assert removed == dict(a=True, b=True, c=True, d=False, e=False, f=False, g=False)
    assert sorted(j["id"] for j in jobs.jobs_for(T1)) == ["d", "e", "f", "g"]  # the others are still listed


def test_a_job_of_another_tenant_cannot_be_removed_or_cancelled():
    from rag import jobs

    add_job("a", T1, "done")
    add_job("b", T1, "queued")
    assert not jobs.delete_job("a", T2) and not jobs.cancel_job("b", T2)
    assert sorted(s for s, _ in state_of(T1)) == ["done", "queued"]


def test_only_a_waiting_job_can_be_cancelled():
    from rag import jobs

    for job_id, state in [("q", "queued"), ("r", "running"), ("d", "done")]:
        add_job(job_id, T1, state)
    assert jobs.cancel_job("q", T1) and not jobs.cancel_job("r", T1) and not jobs.cancel_job("d", T1)
    assert {j["id"]: j["state"] for j in jobs.jobs_for(T1)} == {"q": "cancelling", "r": "running", "d": "done"}
    assert not jobs.cancel_job("q", T1)  # asking twice does nothing


def test_a_cancelled_job_is_skipped_by_the_worker_and_never_downloaded(site, monkeypatch):
    from rag import jobs, worker

    monkeypatch.setattr(worker, "fetch_pdf", lambda *a: pytest.fail("a cancelled job must not be downloaded"))
    add_job("j1", T1, "queued")
    assert jobs.cancel_job("j1", T1)
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})
    assert state_of(T1)[0] == ("cancelled", "cancelled before it started")
    assert jobs.delete_job("j1", T1)  # only now can the line go: the job is really gone from the queue's point of view
    assert not jobs.jobs_for(T1)


def test_a_job_that_was_not_cancelled_runs_as_before(site):
    from rag import worker

    add_job("j1", T1, "queued")
    worker.process({"job_id": "j1", "tenant": T1, "url": f"{site}/doc.pdf"})
    assert state_of(T1)[0][0] == "done"


def test_each_job_line_has_the_right_button(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RAG_TRACING", "0")
    for job_id, state in [("done1", "done"), ("wait1", "queued"), ("run1", "running")]:
        add_job(job_id, T1, state)
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60)
    at.query_params["tenant"] = T1
    at.run()
    assert not at.exception
    buttons = {b.key: b for b in at.sidebar.button}
    assert set(buttons) == {"job-done1", "job-wait1"}  # a running job has no button: it cannot be stopped


def test_clicking_the_x_removes_a_finished_line_and_cancels_a_queued_one(monkeypatch):
    from rag import jobs
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RAG_TRACING", "0")
    add_job("done1", T1, "done")
    add_job("wait1", T1, "queued")
    at = AppTest.from_file("../src/rag/app.py", default_timeout=60)
    at.query_params["tenant"] = T1
    at.run()
    at.sidebar.button(key="job-done1").click().run()
    assert {j["id"] for j in jobs.jobs_for(T1)} == {"wait1"}
    at.sidebar.button(key="job-wait1").click().run()
    assert state_of(T1)[0][0] == "cancelling"
    assert not [b for b in at.sidebar.button if b.key == "job-wait1"]  # it stays listed, with no button until the worker skipped it


def test_fetch_allow_list_trusts_only_the_listed_host(site, tmp_path, pdf, monkeypatch):
    from rag import fetch

    monkeypatch.setattr(fetch, "ALLOW", set())
    with pytest.raises(fetch.Refused):  # the local server is refused until its host is listed
        fetch.fetch_pdf(f"{site}/Q1%20report.pdf", tmp_path / "a" / "a.pdf")
    monkeypatch.setattr(fetch, "ALLOW", {"127.0.0.1"})
    fetch.fetch_pdf(f"{site}/Q1%20report.pdf", tmp_path / "b" / "a.pdf")  # http and a custom port are fine for it
    assert (tmp_path / "b" / "a.pdf").read_bytes() == pdf.read_bytes()
    with pytest.raises(fetch.Refused):  # another name for the same machine is not listed
        fetch.fetch_pdf(site.replace("127.0.0.1", "localhost") + "/Q1%20report.pdf", tmp_path / "c" / "a.pdf")

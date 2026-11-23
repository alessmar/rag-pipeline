"""Run with: uv run streamlit run src/rag/app.py

Embedded in a frontend, pass the company as ?tenant=<uuid>. Without it, a sidebar dropdown lists the tenants."""
import re
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.ask import ask  # noqa: E402
from rag.highlight import render_page  # noqa: E402
from rag.history import add as save_turn, clear as clear_history, load as load_history  # noqa: E402
from rag.jobs import ACTIVE, FINISHED, cancel_job, delete_job, jobs_for  # noqa: E402
from rag.store import default_tenant, ingested_docs, labels, tenant_id, tenants  # noqa: E402
from rag.tracing import setup  # noqa: E402

st.set_page_config(page_title="RAG pipeline", layout="wide")
st.cache_resource(setup)()  # Streamlit reruns this script on every click; trace setup must run once


def pick_tenant() -> str:
    param = st.query_params.get("tenant")
    if param:  # embedded: the URL fixes the tenant, and the other tenants are never listed
        try:
            return tenant_id(param)
        except ValueError:
            st.error("Invalid tenant: expected a UUID in ?tenant=")
            st.stop()
    with st.sidebar:
        known = tenants()
        if not known:
            st.info("Add a company to tenants.json to start.")
            st.stop()
        names = labels()
        first = known.index(default_tenant()) if default_tenant() in known else 0  # only the first selection: after that the choice sticks
        return st.selectbox("Company", known, index=first, format_func=lambda t: names.get(t, t))  # shows the label, returns the UUID


SESSION = re.compile(r"[A-Za-z0-9_-]{8,64}")  # a frontend may pass its own id, for example a user id


def session_id() -> str:
    """One client, one conversation. The id lives in the URL, so a reload or a bookmark opens the same history."""
    sid = st.query_params.get("session")
    if not sid:
        sid = st.query_params["session"] = uuid.uuid4().hex
    elif not SESSION.fullmatch(sid):
        st.error("Invalid session: expected 8 to 64 letters, digits, - or _ in ?session=")
        st.stop()
    return sid


tenant = pick_tenant()
session = session_id()

with st.sidebar:
    st.header("Documents")
    recent = jobs_for(tenant, 8)
    running = any(j["state"] in ACTIVE for j in recent)

    @st.fragment(run_every=3 if running else None)  # polls only while something is in progress
    def show_jobs() -> None:
        current = jobs_for(tenant, 8)
        if running and not any(j["state"] in ACTIVE for j in current):
            st.rerun()  # the last job just finished: redraw the whole page so the counts below are updated
        if current:
            st.caption("Ingestion jobs")
        for j in current:
            left, right = st.columns([5, 1], vertical_alignment="center")
            left.markdown(f"**{j['doc']}**")
            left.caption(j["state"] + (f": {j['detail']}" if j["detail"] else ""))
            if j["state"] == "queued":  # not started yet: the worker can be told to skip it
                icon, hint, action = ":material/close:", "Cancel: the worker will skip it", cancel_job
            elif j["state"] in FINISHED:
                icon, hint, action = ":material/close:", "Remove from this list", delete_job
            else:  # running, retrying or already being cancelled: it cannot be stopped or removed now, so no button
                continue
            if right.button("", key=f"job-{j['id']}", icon=icon, help=hint, type="tertiary"):
                action(j["id"], tenant)
                st.rerun()
        if any(j["state"] == "queued" for j in current):
            st.caption("Waiting for the worker. Start it with `uv run rag worker`.")

    show_jobs()

    docs = ingested_docs(tenant)
    st.metric("Ingested PDFs", len(docs))
    st.caption(f"{sum(docs.values())} chunks in total")
    with st.expander("Show list"):
        st.dataframe([{"pdf": d, "chunks": n} for d, n in sorted(docs.items())], hide_index=True)


def show_answer(m: dict) -> None:
    st.markdown(m["answer"])
    for s in m["sources"]:
        with st.expander(f"[{s['n']}] {s['doc']}, p.{s['page']}"):
            st.caption(s["heading"])
            for q in s["quotes"]:
                st.markdown(f"> {q['text']}")
            if Path(s["path"]).exists():
                st.image(render_page(s["path"], s["page"], s["items"]))


key = f"history:{tenant}:{session}"
if key not in st.session_state:  # a new session, a reload or a restart: start from what was saved
    st.session_state[key] = load_history(tenant, session)
history = st.session_state[key]
if history:
    with st.sidebar:
        if st.button("Clear history", icon=":material/close:", type="tertiary"):
            clear_history(tenant, session)
            history.clear()
            st.rerun()
for m in history:
    with st.chat_message("user"):
        st.write(m["question"])
    with st.chat_message("assistant"):
        show_answer(m)
if question := st.chat_input("Ask about your documents", disabled=not docs):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("Searching and answering"):
        m = {"question": question, **ask(question, tenant, history=history)}
        show_answer(m)
    history.append(m)
    save_turn(tenant, session, m)
if not docs:
    st.info("No documents yet. They arrive through the queue: queue a PDF URL with scripts/produce.py.")

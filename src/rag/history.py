"""Chat history: every answered question is kept per company and per session, so a reload or a restart does not lose
the conversation. A session is one client: its id comes from the page URL (?session=...)."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DB = os.environ.get("RAG_HISTORY_DB", "data/history.sqlite")
SHOWN = 20  # turns loaded when the page opens: each one draws its source pages again


def _conn() -> sqlite3.Connection:
    Path(HISTORY_DB).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(HISTORY_DB, timeout=30)
    c.execute(
        "create table if not exists turns (id integer primary key autoincrement, tenant text, session text default '',"
        " created text, turn text)"
    )
    if "session" not in [r[1] for r in c.execute("pragma table_info(turns)")]:  # a table from before sessions existed
        c.execute("alter table turns add column session text default ''")
    c.execute("create index if not exists turns_session on turns (tenant, session, id)")
    return c


def load(tenant: str, session: str, limit: int = SHOWN) -> list[dict]:
    """The last `limit` turns of this session in this company, oldest first."""
    with closing(_conn()) as c:
        rows = c.execute(
            "select turn from turns where tenant = ? and session = ? order by id desc limit ?", (tenant, session, limit)
        ).fetchall()
    return [json.loads(r[0]) for r in reversed(rows)]


def add(tenant: str, session: str, turn: dict) -> None:
    turn = {k: v for k, v in turn.items() if k != "chunks"}  # the retrieved text is large and the page does not need it
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_conn()) as c, c:
        c.execute(
            "insert into turns (tenant, session, created, turn) values (?, ?, ?, ?)", (tenant, session, now, json.dumps(turn))
        )


def clear(tenant: str, session: str) -> None:
    with closing(_conn()) as c, c:
        c.execute("delete from turns where tenant = ? and session = ?", (tenant, session))

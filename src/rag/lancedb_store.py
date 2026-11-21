"""LanceDB implementation of the VectorStore interface in store.py. One database folder per tenant.

Keyword search uses two columns, text_it and text_en, each with an index that knows its language (stemming and stop
words). A chunk goes to the column of the language it is written in, or into both when that cannot be told. A search
runs the vector query and both keyword queries, and merges the three rankings."""
import os
from collections import Counter
from pathlib import Path

import lancedb
from lancedb.index import FTS

from rag.language import detect
from rag.store import tenant_id

DB_DIR = os.environ.get("RAG_DB_DIR", "data/lancedb")
TABLE = "chunks"
KEYS = ("id", "doc", "path", "page", "heading", "text", "items")
RRF_K = 60  # reciprocal rank fusion: a chunk scores weight / (RRF_K + rank) in each ranking it appears in
# Measured on ten questions: each keyword ranking at half weight (together they weigh as much as the vector ranking),
# and rankings no deeper than the 8 results the app uses. Deeper rankings, or full-weight keywords, let weak matches pile up.
KEYWORD_WEIGHT = 0.5
MIN_DEPTH = 8


def _with_language_columns(rows: list[dict]) -> list[dict]:
    """A document is in one language, so every chunk takes the language most of its document's chunks were detected
    in. That also fixes chunks that are hard to judge alone (citation lists, names, numbers). A document with no
    confident chunk at all stays unsure, and its chunks are searchable in both columns."""
    votes: dict[str, Counter] = {}
    for r in rows:
        if lang := detect(r["text"]):
            votes.setdefault(r["doc"], Counter())[lang] += 1
    # shortcut: a document that really mixes two languages gets only the main one, split it per chunk if that appears
    main = {doc: c.most_common(1)[0][0] for doc, c in votes.items()}
    out = []
    for r in rows:
        lang = main.get(r["doc"], "")
        out.append({**r, "lang": lang, "text_it": r["text"] if lang != "en" else "", "text_en": r["text"] if lang != "it" else ""})
    return out


class LanceStore:
    def _exists(self, tenant: str) -> bool:
        # checks the folder instead of connecting, so asking about an unknown tenant creates nothing
        return (Path(DB_DIR) / tenant_id(tenant) / f"{TABLE}.lance").exists()

    def _db(self, tenant: str):
        return lancedb.connect(str(Path(DB_DIR) / tenant_id(tenant)))

    def _table(self, tenant: str):
        return self._db(tenant).open_table(TABLE)

    @staticmethod
    def _index(t) -> None:
        # with_position: a question in "quotes" is an exact-phrase search, which needs to know where each word sits
        t.create_index("text_it", config=FTS(language="Italian", stem=True, remove_stop_words=True, with_position=True), replace=True)
        t.create_index("text_en", config=FTS(language="English", stem=True, remove_stop_words=True, with_position=True), replace=True)

    def reindex(self, tenant: str) -> Counter:
        """Recompute every chunk's language columns and rebuild both keyword indexes. It also upgrades a table made
        before these columns existed. Returns how many chunks ended up as it, en, or unsure."""
        if not self._exists(tenant):
            return Counter()
        t = self._table(tenant)
        rows = _with_language_columns(t.search().limit(t.count_rows()).to_arrow().to_pylist())
        self._index(self._db(tenant).create_table(TABLE, rows, mode="overwrite"))
        return Counter(r["lang"] or "unsure" for r in rows)

    def replace_doc(self, tenant: str, doc: str, rows: list[dict]) -> None:
        rows = _with_language_columns(rows)
        if self._exists(tenant):
            if "text_it" not in self._table(tenant).schema.names:
                self.reindex(tenant)  # an older table: add the language columns to what is already there
            t = self._table(tenant)
            t.delete(f"doc = '{doc.replace(chr(39), chr(39) * 2)}'")
            t.add(rows)
        else:
            t = self._db(tenant).create_table(TABLE, rows)
        self._index(t)

    def search(self, tenant: str, text: str, vector: list[float], k: int) -> list[dict]:
        if not self._exists(tenant):
            raise LookupError(f"tenant {tenant} has no documents")
        t = self._table(tenant)
        if "text_it" not in t.schema.names:
            raise RuntimeError(f"this table predates the language columns: run  uv run rag reindex --tenant {tenant}")
        depth = max(k, MIN_DEPTH)  # how deep each of the three rankings goes before they are merged
        rankings = [
            (1.0, t.search(vector, query_type="vector").limit(depth).to_list()),
            (KEYWORD_WEIGHT, t.search(text, query_type="fts", fts_columns="text_it").limit(depth).to_list()),
            (KEYWORD_WEIGHT, t.search(text, query_type="fts", fts_columns="text_en").limit(depth).to_list()),
        ]
        score, rows = {}, {}
        for weight, ranking in rankings:
            for rank, r in enumerate(ranking, 1):
                score[r["id"]] = score.get(r["id"], 0) + weight / (RRF_K + rank)
                rows[r["id"]] = r
        best = sorted(score, key=score.get, reverse=True)[:k]
        return [{**{key: rows[i][key] for key in KEYS}, "score": score[i]} for i in best]

    def docs(self, tenant: str) -> Counter:
        if not self._exists(tenant):
            return Counter()
        t = self._table(tenant)
        return Counter(t.search().select(["doc"]).limit(t.count_rows()).to_arrow().column("doc").to_pylist())

    def tenants(self) -> list[str]:
        root = Path(DB_DIR)
        found = []
        for p in root.iterdir() if root.exists() else []:
            try:
                if p.is_dir() and tenant_id(p.name) == p.name:
                    found.append(p.name)
            except ValueError:
                pass  # not a tenant folder
        return sorted(found)

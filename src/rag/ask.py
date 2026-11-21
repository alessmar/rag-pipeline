import json
import os
import re
import time

from openai import OpenAI
from opentelemetry import trace
from pydantic import BaseModel

from rag.search import search

tracer = trace.get_tracer("rag")

MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")  # none, low, medium, high, xhigh, max

INSTRUCTIONS = """Answer the question using ONLY the numbered excerpts provided.
The excerpts may be in a different language than the question. Write the answer in the language of the QUESTION,
not the language of the excerpts. Quotes in the citations stay verbatim, in the excerpt's own language.
First set question_language to the language the question is written in.
Put the excerpt number in square brackets after each claim, e.g. "The limit is 5 kg [2]".
For every number you use, add a citation with an exact, verbatim quote copied from that excerpt.
If the excerpts do not contain the answer, say so and return no citations."""


class Citation(BaseModel):
    chunk: int
    quote: str


class Answer(BaseModel):
    question_language: str  # first on purpose: deciding it before writing stops the excerpts' language taking over
    answer: str
    citations: list[Citation]


CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [3], [3][5] and [3, 5]


def _renumber(answer: str, sources: dict[int, dict]) -> tuple[str, list[dict]]:
    """The model cites by retrieval position (e.g. [3] and [5]); show readers 1, 2, 3 in order of appearance."""
    order = []
    for m in CITE.finditer(answer):
        for n in map(int, re.findall(r"\d+", m.group(1))):
            if n in sources and n not in order:
                order.append(n)
    order += [n for n in sources if n not in order]  # cited with a quote but never marked in the text
    new = {old: i for i, old in enumerate(order, 1)}

    def fix(m: re.Match) -> str:
        nums = [new[n] for n in map(int, re.findall(r"\d+", m.group(1))) if n in new]
        return "".join(f"[{n}]" for n in dict.fromkeys(nums))  # markers without a valid source vanish

    return CITE.sub(fix, answer), [{**sources[old], "n": new[old]} for old in order]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def ask(question: str, tenant: str, k: int = 8) -> dict:
    with tracer.start_as_current_span("ask") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("input.value", question)
        span.set_attribute("metadata", json.dumps({"tenant": tenant}))
        result = _ask(question, tenant, k)
        span.set_attribute("output.value", result["answer"])
        result["trace_id"] = format(span.get_span_context().trace_id, "032x")
        return result


def _ask(question: str, tenant: str, k: int) -> dict:
    t0 = time.perf_counter()
    chunks = search(question, tenant, k)
    excerpts = "\n\n".join(f"[{i}] ({c['doc']}, p.{c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    resp = OpenAI().responses.parse(
        model=MODEL,
        input=[
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
        ],
        text_format=Answer,
        reasoning={"effort": REASONING_EFFORT},  # reasoning tokens are billed as output
    )
    out = resp.output_parsed
    sources: dict[int, dict] = {}  # one entry per cited excerpt, even if the model cites it several times
    for c in out.citations:
        if not 1 <= c.chunk <= len(chunks):
            continue  # the model cited a number that was never retrieved
        ch = chunks[c.chunk - 1]
        src = sources.setdefault(
            c.chunk,
            {
                "n": c.chunk,
                "doc": ch["doc"],
                "page": ch["page"],
                "heading": ch["heading"],
                "path": ch["path"],
                "items": json.loads(ch["items"]),
                "quotes": [],
            },
        )
        src["quotes"].append({"text": c.quote, "verified": _norm(c.quote) in _norm(ch["text"])})
    answer, numbered = _renumber(out.answer, sources)
    return {"answer": answer, "sources": numbered, "chunks": chunks, "seconds": time.perf_counter() - t0}

import json
from functools import lru_cache
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from rag.embed import MODEL, embed_passages
from rag.store import replace_doc


class NoTextError(ValueError):
    """The PDF has no extractable text. Retrying cannot help."""


@lru_cache
def _converter() -> DocumentConverter:
    return DocumentConverter()  # loads the layout models once, not once per PDF


@lru_cache
def _chunker() -> HybridChunker:
    # chunk size follows the embedding model's limit (512 tokens)
    tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(MODEL), max_tokens=512)
    return HybridChunker(tokenizer=tok)


def chunk_pdf(pdf: Path, final: Path | None = None) -> list[dict]:
    """Parse with Docling, then chunk. Each chunk keeps page, paragraph refs and bbox.
    `final` is where the PDF will live when it is not read from there (it is read from a staging file first)."""
    final = final or pdf
    doc = _converter().convert(pdf).document
    rows = []
    for i, ch in enumerate(_chunker().chunk(doc)):
        items = [
            {"ref": it.self_ref, "page": p.page_no, "bbox": p.bbox.model_dump(mode="json")}
            for it in ch.meta.doc_items
            for p in it.prov
        ]
        if not items:
            continue
        rows.append(
            {
                "id": f"{final.stem}:{i}",
                "doc": final.name,
                "path": str(final.resolve()),
                "page": items[0]["page"],
                "heading": " > ".join(ch.meta.headings or []),
                "text": ch.text,
                "items": json.dumps(items),
            }
        )
    return rows


def ingest(pdf: Path, tenant: str, final: Path | None = None) -> int:
    final = final or pdf
    rows = chunk_pdf(pdf, final)
    if not rows:
        raise NoTextError(f"no text extracted from {pdf} (scanned PDF? enable OCR)")
    vectors = embed_passages([r["text"] for r in rows])
    for r, v in zip(rows, vectors):
        r["vector"] = v
    replace_doc(tenant, final.name, rows)  # re-ingesting a PDF replaces its chunks
    return len(rows)

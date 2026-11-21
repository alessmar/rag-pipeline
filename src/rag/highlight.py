import pymupdf


def render_page(path: str, page: int, items: list[dict]) -> bytes:
    """PNG of one PDF page with the cited paragraphs highlighted (bboxes come from Docling)."""
    doc = pymupdf.open(path)
    pg = doc[page - 1]
    h = pg.rect.height
    for it in items:
        if it["page"] != page:
            continue
        b = it["bbox"]
        # Docling boxes are usually measured from the bottom-left corner; PDF rendering is top-left
        top, bottom = (h - b["t"], h - b["b"]) if b["coord_origin"] == "BOTTOMLEFT" else (b["t"], b["b"])
        pg.add_highlight_annot(pymupdf.Rect(b["l"], min(top, bottom), b["r"], max(top, bottom)))
    return pg.get_pixmap(dpi=110).tobytes("png")

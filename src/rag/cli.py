import argparse
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    from rag.store import tenant_id

    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="ingest PDF files and keep a copy in the tenant's PDF folder, as a queued job does")
    i.add_argument("--tenant", required=True, type=tenant_id, help="company UUID")
    i.add_argument("pdfs", nargs="+", type=Path)
    r = sub.add_parser("reindex", help="rebuild a tenant's keyword indexes; stop the worker first")
    r.add_argument("--tenant", required=True, type=tenant_id, help="company UUID")
    sub.add_parser("worker", help="process queued ingestion jobs from Redis, one at a time")
    a = sub.add_parser("ask")
    a.add_argument("--tenant", required=True, type=tenant_id, help="company UUID")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=8)
    args = p.parse_args()

    if args.cmd == "ingest":
        import os
        import shutil
        import uuid

        from rag.ingest import ingest
        from rag.store import pdf_dir

        for pdf in args.pdfs:
            # Same as a queued job: the app reads the page from the tenant's folder, so the PDF must live there.
            # It is copied to a hidden file first and takes its real name only after ingestion worked.
            target = pdf_dir(args.tenant) / pdf.name
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".staging-{uuid.uuid4().hex}.pdf")
            try:
                shutil.copyfile(pdf, staging)
                chunks = ingest(staging, args.tenant, target)
                os.replace(staging, target)
            finally:
                staging.unlink(missing_ok=True)
            print(f"{pdf.name}: {chunks} chunks")
    elif args.cmd == "reindex":
        from rag.store import reindex

        counts = reindex(args.tenant)
        print(f"{sum(counts.values())} chunks reindexed: {dict(counts)}" if counts else "no documents for this tenant")
    elif args.cmd == "worker":
        from rag.celery_app import app

        # solo pool: one job at a time, in this process, so the models stay loaded and LanceDB has a single writer
        app.worker_main(["worker", "--pool=solo", "--concurrency=1", "--loglevel=INFO"])
    else:
        from rag.ask import ask
        from rag.tracing import setup

        setup()
        r = ask(args.question, args.tenant, args.k)
        print(r["answer"], "\n")
        for s in r["sources"]:
            print(f"[{s['n']}] {s['doc']}, p.{s['page']}, {', '.join(it['ref'] for it in s['items'])}")
            for q in s["quotes"]:
                flag = "" if q["verified"] else "  [QUOTE NOT FOUND IN SOURCE]"
                print(f"    \"{q['text']}\"{flag}")


if __name__ == "__main__":
    main()

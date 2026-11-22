"""Throwaway producer: puts PDF URLs on the queue so the worker picks them up.

  uv run python scripts/produce.py --tenant <company uuid> https://host/a.pdf https://host/b.pdf

It needs the same .env as the app (RAG_REDIS_URL). The job is checked and downloaded by the worker, not here."""
import argparse

from dotenv import load_dotenv

load_dotenv()  # before rag is imported: its settings are read from the environment on import

from rag.jobs import submit_url  # noqa: E402
from rag.store import tenant_id  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tenant", required=True, type=tenant_id, help="company UUID")
    p.add_argument("urls", nargs="+")
    args = p.parse_args(argv)
    for url in args.urls:
        print(f"{url} -> job {submit_url(args.tenant, url)}")


if __name__ == "__main__":
    main()

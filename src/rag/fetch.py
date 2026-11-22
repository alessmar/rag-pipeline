"""Download a PDF from a URL, for the worker. The URL comes from a client, so everything here treats it as hostile:
only HTTPS on port 443 to public addresses (unless the host is in RAG_FETCH_ALLOW), redirects followed by hand and re-checked, and size and time capped."""
import hashlib
import ipaddress
import os
import socket
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpcore

MAX_PDF_BYTES = int(os.environ.get("RAG_MAX_PDF_MB", "50")) * (1 << 20)
ALLOW = {h.strip().lower() for h in os.environ.get("RAG_FETCH_ALLOW", "").split(",") if h.strip()}  # host names the operator trusts, e.g. an internal MinIO
MAX_REDIRECTS = 3
DEADLINE_SECONDS = 120
TIMEOUTS = {"connect": 10, "read": 30, "write": 10, "pool": 10}


class Refused(Exception):
    """The URL can never work (blocked address, not a PDF, too large, HTTP 4xx). Retrying cannot help."""


class Transient(Exception):
    """It may work later (timeout, connection error, HTTP 5xx or 429)."""


def doc_name(url: str) -> str:
    """The document name for a URL: its last path segment if that is a usable .pdf name, else a hash of the URL."""
    name = Path(unquote(urlsplit(url).path)).name
    if not name.lower().endswith(".pdf") or name.startswith(".") or len(name) > 200 or any(ord(c) < 32 for c in name):
        name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".pdf"
    return name


def _allowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 6 and ip.ipv4_mapped:  # ::ffff:10.0.0.1 is really 10.0.0.1
        ip = ip.ipv4_mapped
    return ip.is_global  # private, loopback, link-local (cloud metadata) and reserved ranges are not global


def _public_address(host: str, port: int) -> str:
    trusted = host.lower() in ALLOW
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise Transient(f"cannot resolve {host}: {e}")
    if not trusted and not all(_allowed(ipaddress.ip_address(i[4][0].split("%")[0])) for i in infos):
        raise Refused(f"{host} resolves to a non-public address")
    return infos[0][4][0]


class _Guarded(httpcore.SyncBackend):
    """Resolves the name itself and connects to the address it checked, so the check and the connection cannot
    disagree (DNS rebinding). TLS still verifies the certificate against the original host name."""

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return super().connect_tcp(_public_address(host, port), port, timeout=timeout,
                                   local_address=local_address, socket_options=socket_options)


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    loose = (parts.hostname or "").lower() in ALLOW  # an allowed host may use http and any port
    if parts.scheme not in (("https", "http") if loose else ("https",)):
        raise Refused("only https URLs are accepted")
    if not parts.hostname or parts.username or parts.password:
        raise Refused("the URL must have a host and no embedded credentials")
    if not loose and parts.port not in (None, 443):
        raise Refused("only the default https port is accepted")


def fetch_pdf(url: str, dest: Path) -> None:
    """Download the PDF to the file `dest`, which only appears once the whole download passed every check.
    Raises Refused or Transient, and then leaves nothing behind."""
    deadline = time.monotonic() + DEADLINE_SECONDS
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = url
    part = dest.with_name(f".incoming-{uuid.uuid4().hex}.part")
    try:
        with httpcore.ConnectionPool(network_backend=_Guarded(), http2=False) as pool:
            for _ in range(MAX_REDIRECTS + 1):
                _check_url(current)
                with pool.stream("GET", current, headers={"User-Agent": "rag-fetch", "Accept": "application/pdf"},
                                 extensions={"timeout": TIMEOUTS}) as r:
                    if r.status in (301, 302, 303, 307, 308):
                        where = dict((k.lower(), v) for k, v in r.headers).get(b"location")
                        if not where:
                            raise Refused("redirect without a Location header")
                        current = urljoin(current, where.decode("latin-1"))  # re-checked at the top of the loop
                        continue
                    if r.status == 429 or r.status >= 500:
                        raise Transient(f"HTTP {r.status}")
                    if r.status != 200:
                        raise Refused(f"HTTP {r.status}")
                    size = 0
                    with open(part, "wb") as out:
                        for chunk in r.iter_stream():
                            if size == 0 and not chunk.startswith(b"%PDF-"):  # the content type is not trusted
                                raise Refused("the URL does not serve a PDF")
                            size += len(chunk)
                            if size > MAX_PDF_BYTES:
                                raise Refused(f"file larger than {MAX_PDF_BYTES >> 20} MB")
                            if time.monotonic() > deadline:
                                raise Transient("download took too long")
                            out.write(chunk)
                    if size == 0:
                        raise Refused("empty response")
                    os.replace(part, dest)
                    return
            raise Refused("too many redirects")
    except httpcore.UnsupportedProtocol as e:
        raise Refused(str(e))
    except (httpcore.TimeoutException, httpcore.NetworkError, httpcore.ProtocolError) as e:
        raise Transient(f"{type(e).__name__}: {e}")
    finally:
        part.unlink(missing_ok=True)

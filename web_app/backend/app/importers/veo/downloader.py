"""Streaming downloader for user-confirmed, authorized Veo resources."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.request import Request, build_opener, HTTPRedirectHandler, HTTPCookieProcessor
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from urllib.parse import urlsplit
import ipaddress
import socket


def public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTPS media URLs are supported")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("Private network resources are not supported")


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def authorized_opener(cookies: list[dict]):
    jar = CookieJar(policy=DefaultCookiePolicy(strict_ns_domain=DefaultCookiePolicy.DomainStrictNonDomain))
    for value in cookies:
        domain = value["domain"]
        jar.set_cookie(Cookie(0, value["name"], value["value"], None, False, domain, domain.startswith("."), domain.startswith("."), value.get("path", "/"), True, value.get("secure", True), int(value["expires"]) if value.get("expires", -1) > 0 else None, False, None, None, {}))
    return build_opener(PublicRedirect(), HTTPCookieProcessor(jar))


def download_stream(url: str, destination: Path, progress: Callable[[int, int | None], None], cancelled: Callable[[], bool], cookies: list[dict] | None = None) -> None:
    if cancelled():
        raise InterruptedError("Import cancelled.")
    public_url(url)
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing imported asset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": "VeoMatchLab local importer"})
    downloaded = 0
    created_partial = False
    try:
        with authorized_opener(cookies or []).open(request, timeout=20) as response, partial.open("xb") as target:
            created_partial = True
            if response.status != 200:
                raise ValueError("The server did not return a complete asset")
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None
            while True:
                if cancelled():
                    raise InterruptedError("Import cancelled.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                progress(downloaded, expected)
            if expected is not None and downloaded != expected:
                raise ValueError("Incomplete download")
            if cancelled():
                raise InterruptedError()
        partial.replace(destination)
    except BaseException:
        # A previous interrupted run may own this path. Failed exclusive creation
        # must never delete somebody else's recoverable partial download.
        if created_partial:
            partial.unlink(missing_ok=True)
        raise

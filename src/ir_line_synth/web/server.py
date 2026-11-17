"""Local HTTP server of the editor: the page, its files and the JSON API."""

import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STATIC = Path(__file__).parent / "static"
# Fixed types of the page files. mimetypes takes them from the Windows registry, where
# .js is text/plain on some machines; with nosniff a browser refuses to run such a script.
STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
STATIC_FILES = {path.name for path in STATIC.iterdir() if path.suffix in STATIC_TYPES}
API_PREFIX = "/api/synthetic/"
BODY_LIMIT = 512 * 1024
# Saving generated scenes sends every selected scene description in one request.
BATCH_BODY_LIMIT = 4 * 1024 * 1024
BATCH_SAVE = API_PREFIX + "generated-save"
# The autogenerator sends its object pool; generated objects carry their motion parameters.
BATCH_REQUESTS = {BATCH_SAVE, API_PREFIX + "generate"}
# A tape description lists every object placed on it.
BATCH_REQUESTS |= {API_PREFIX + name for name in ("tape-generate", "tape-preview", "tape-save")}
REQUEST_ERRORS = (ValueError, OSError, RuntimeError, KeyError, TypeError)


def make_server(views, port=8009):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, allow_nan=False).encode(),
                            "application/json; charset=utf-8", status)

        def do_GET(self):
            url = urlparse(self.path)
            name = "index.html" if url.path == "/" else url.path[1:]
            try:
                if url.path.startswith(API_PREFIX):
                    self.send_json(views.get(url.path, parse_qs(url.query)))
                elif name in STATIC_FILES:
                    path = STATIC / name
                    self.send_bytes(path.read_bytes(), STATIC_TYPES[path.suffix] + "; charset=utf-8")
                else:
                    self.send_json({"error": "Not found"}, 404)
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            # Only same-origin browser requests or explicit local clients can write files.
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            if origin and origin != f"http://{host}":
                self.send_json({"error": "Same-origin request required"}, 403)
                return
            port = self.server.server_port
            # The editor reads local folders and writes files, so it must not accept DNS-rebinding hosts.
            if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                self.send_json({"error": "Localhost Host header required"}, 403)
                return
            url = urlparse(self.path)
            if not url.path.startswith(API_PREFIX):
                self.send_json({"error": "Not found"}, 404)
                return
            limit = BATCH_BODY_LIMIT if url.path in BATCH_REQUESTS else BODY_LIMIT
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= limit:
                    raise ValueError(f"Expected a JSON body of at most {limit // 1024} KiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                self.send_json(views.post(url.path, body))
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, format, *args):
            # Every click calls the API; errors are shown in the page.
            pass

    return EditorServer(("127.0.0.1", port), Handler)


class EditorServer(ThreadingHTTPServer):
    """One editor per port.

    On Windows the default SO_REUSEADDR lets a second process bind a port that
    is already served. The old editor then keeps answering, and a restarted
    one seems to have no effect. An exclusive bind makes the second start fail.
    """

    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

"""Small independent HTTP health server that cannot block analytics."""

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    ready: bool
    configuration_loaded: bool
    cameras_ready: bool
    models_ready: bool
    details: dict[str, str]


class EdgeHealthServer:
    def __init__(
        self,
        readiness: Callable[[], ReadinessSnapshot],
        metrics: Callable[[], str],
        host: str = "127.0.0.1",
        port: int = 8090,
    ) -> None:
        self._readiness = readiness
        self._metrics = metrics
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        readiness = self._readiness
        metrics = self._metrics

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health/live":
                    self._json(HTTPStatus.OK, {"status": "alive"})
                    return
                if self.path == "/health/ready":
                    snapshot = readiness()
                    code = HTTPStatus.OK if snapshot.ready else HTTPStatus.SERVICE_UNAVAILABLE
                    self._json(code, asdict(snapshot))
                    return
                if self.path == "/metrics":
                    body = metrics().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"detail": "not found"})

            def _json(self, status: HTTPStatus, payload: object) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        return Handler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="edge-health-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)

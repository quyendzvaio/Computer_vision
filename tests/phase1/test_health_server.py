import json
import urllib.error
import urllib.request

from edge_runtime.health import EdgeHealthServer, MetricsRegistry, ReadinessSnapshot


def test_health_server_exposes_live_ready_and_metrics() -> None:
    metrics = MetricsRegistry()
    metrics.set_gauge("edge_test_value", 2)
    server = EdgeHealthServer(
        lambda: ReadinessSnapshot(False, True, False, False, {"reason": "test"}),
        metrics.render_prometheus,
        port=0,
    )
    server.start()
    host, port = server.address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health/live") as response:
            assert response.status == 200
        try:
            urllib.request.urlopen(f"http://{host}:{port}/health/ready")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
            body = json.loads(exc.read())
            assert body["ready"] is False
        with urllib.request.urlopen(f"http://{host}:{port}/metrics") as response:
            assert "edge_test_value 2.0" in response.read().decode()
    finally:
        server.stop()

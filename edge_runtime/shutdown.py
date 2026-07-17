"""Cross-platform signal handling for graceful edge shutdown."""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def shutdown_signals() -> Iterator[threading.Event]:
    stop = threading.Event()
    previous: dict[signal.Signals, object] = {}

    def handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handle_signal)
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

"""Phase-1 worker lifecycle placeholder; no event handlers are enabled yet."""

import logging
import signal
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("event-worker")


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.warning("Phase-1 event worker started without notification handlers")
    while not stop.wait(5.0):
        pass


if __name__ == "__main__":
    main()

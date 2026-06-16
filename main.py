#!/usr/bin/env python3
"""CV Safety Monitor v2 — Main entry point.

Delegates to GPU machine app (PyQt5 + QThread architecture).
Edge device runs separately (edge/sender.py).
"""
import asyncio
import multiprocessing
import signal
import sys
import threading


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_dashboard(alert_pipeline):
    """Start FastAPI dashboard server."""
    import uvicorn
    from dashboard.server import app, ws_manager

    # Wire ws_manager into the dispatcher
    alert_pipeline.dispatcher._ws_manager = ws_manager

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def start_mqtt_remote_cameras(config, ws_manager, scheduler):
    """Thread: connect MQTT subscriber for remote cameras, feed scheduler."""
    import time
    from inference.mqtt_subscriber import MQTTSubscriber

    remote_camera_ids = []
    for cam in config.get("cameras", []):
        if not str(cam.get("source", "")).isdigit():
            remote_camera_ids.append(cam["id"])

    if not remote_camera_ids:
        return

    broker = config.get("mqtt", {}).get("broker", "localhost")
    port = config.get("mqtt", {}).get("port", 1883)

    sub = MQTTSubscriber(broker=broker, port=port)
    try:
        sub.connect(
            on_frame=lambda cid, jpeg: scheduler.add_frame(cid, jpeg),
        )
        print(f"[Main] MQTT subscriber started for {remote_camera_ids}")
    except ConnectionError:
        print(f"[Main] Warning: Could not connect to MQTT broker, skipping remote cameras")
        return

    # Keep alive
    while not getattr(start_mqtt_remote_cameras, "_stop", False):
        time.sleep(1)

    sub.disconnect()


async def main_async(config_path: str = "edge/config.yaml"):
    """Async main — wires everything together."""
    config = load_config(config_path)

    stop_event = threading.Event()       # For P1 thread
    mp_stop = multiprocessing.Event()    # For P2 process — threading.Event can't cross processes

    # Docker sends SIGTERM for graceful shutdown — stop both P1 and P2
    loop = asyncio.get_running_loop()
    try:
        def _sigterm():
            stop_event.set()
            mp_stop.set()
        loop.add_signal_handler(signal.SIGTERM, _sigterm)
    except NotImplementedError:
        pass  # Signal handlers not supported on this platform

    # --- Classify cameras ---
    local_camera_ids = []
    remote_camera_ids = []
    for cam in config.get("cameras", []):
        source = str(cam.get("source", ""))
        cid = cam.get("id", "")
        if source.isdigit():
            local_camera_ids.append(cid)
        else:
            remote_camera_ids.append(cid)

    # --- Alert Pipeline (runs in P1) ---
    import alert.db as db_module
    db_module.init_db()
    from alert import AlertPipeline
    from alert.roi_matcher import ROIMatcher
    from alert.classifier import ViolationClassifier
    from alert.cooldown import CooldownManager
    from alert.dispatcher import Dispatcher

    roi = ROIMatcher(db=db_module)
    import dashboard.server as dashboard_server
    dashboard_server._roi_matcher = roi

    pipeline = AlertPipeline(
        roi_matcher=roi,
        classifier=ViolationClassifier(),
        cooldown=CooldownManager(),
        dispatcher=Dispatcher(db=db_module),
    )

    # --- Remote cameras via MQTT (thread, if any) ---
    from inference.scheduler import Scheduler
    remote_scheduler = Scheduler()

    mqtt_thread = None
    if remote_camera_ids:
        for cid in remote_camera_ids:
            remote_scheduler.register_camera(cid)
        mqtt_thread = threading.Thread(
            target=start_mqtt_remote_cameras,
            args=(config, dashboard_server.ws_manager, remote_scheduler),
            daemon=True,
        )
        mqtt_thread.start()

    # --- Start P2: Inference subprocess ---
    from shared.memory import FrameBuffer
    from edge.capture_loop import CaptureDisplay

    # Create FrameBuffers for local cameras before starting P2
    max_w = config.get("frame", {}).get("max_width", 1920)
    max_h = config.get("frame", {}).get("max_height", 1080)
    buffers = {}
    for cid in local_camera_ids:
        buf = FrameBuffer(cid, max_width=max_w, max_height=max_h)
        buffers[cid] = buf

    p2 = None
    if local_camera_ids:
        from inference.inference_worker import inference_worker

        p2 = multiprocessing.Process(
            target=inference_worker,
            args=(config, mp_stop),
            daemon=True,
        )
        p2.start()
        print(f"[Main] P2 inference process started (PID {p2.pid})")

    # --- Start P1: CaptureDisplay (thread in main process) ---
    p1_thread = None
    if local_camera_ids:
        p1 = CaptureDisplay(
            config=config,
            stop_event=stop_event,
            loop=loop,
            alert_pipeline=pipeline,
            ws_manager=dashboard_server.ws_manager,
            buffers=buffers,
        )
        p1_thread = threading.Thread(target=p1.run, daemon=True)
        p1_thread.start()
        print(f"[Main] P1 capture-display started for {local_camera_ids}")

    # --- MQTT remote frames: poll scheduler and broadcast preview ---
    async def poll_remote_frames():
        import base64
        import json
        while not stop_event.is_set():
            result = remote_scheduler.poll()
            if result is None:
                await asyncio.sleep(0.01)
                continue
            camera_id, jpeg_bytes = result

            # Remote frames: just broadcast preview (no model in P1 for detection)
            b64 = base64.b64encode(jpeg_bytes).decode()
            await dashboard_server.ws_manager.broadcast_async(json.dumps({
                "type": "preview",
                "camera_id": camera_id,
                "frame_base64": b64,
            }))

    # --- Start dashboard ---
    dashboard_task = asyncio.create_task(run_dashboard(pipeline))

    # Poll remote frames task
    remote_task = None
    if remote_camera_ids:
        remote_task = asyncio.create_task(poll_remote_frames())

    print("[Main] All subsystems started. Running...")

    try:
        # Wire stop events so P2 stops when P1 does
        def _on_stop():
            stop_event.set()
            mp_stop.set()

        # Wait for P1 thread (capture) if it exists, else just keep alive
        if p1_thread:
            while p1_thread.is_alive() and not stop_event.is_set():
                await asyncio.sleep(1)
        else:
            while not stop_event.is_set():
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        print("[Main] Shutting down...")
        _on_stop()

        # Stop P2
        if p2 and p2.is_alive():
            p2.join(timeout=5)
            if p2.is_alive():
                p2.terminate()

        if remote_task:
            remote_task.cancel()
            try:
                await remote_task
            except asyncio.CancelledError:
                pass

        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass

        # Cleanup shared memory
        for buf in buffers.values():
            buf.close()
        for buf in buffers.values():
            buf.unlink()

        if mqtt_thread:
            start_mqtt_remote_cameras._stop = True

        print("[Main] Shutdown complete")


def main():
    """Entry point — handles signals and runs the async main."""
    # Required for multiprocessing shared memory on some platforms
    multiprocessing.set_start_method("spawn", force=True)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "edge/config.yaml"
    try:
        asyncio.run(main_async(config_path))
    except KeyboardInterrupt:
        pass
    print("[Main] Goodbye.")


if __name__ == "__main__":
    main()

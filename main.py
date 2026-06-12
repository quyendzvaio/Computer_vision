#!/usr/bin/env python3
"""CV Safety Monitor — Main entry point.
Orchestrates all subsystems: Edge, Inference, Alert, Dashboard."""

import sys
import asyncio
import signal
import threading
from pathlib import Path

import yaml


def load_config(config_path: str = "edge/config.yaml") -> dict:
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


async def run_inference_loop(inference_engine, alert_pipeline, scheduler, stop_event):
    """Main inference loop: poll scheduler -> detect -> alert pipeline."""
    import base64
    import json
    from dashboard.server import ws_manager

    print("[Main] Inference loop started")
    while not stop_event.is_set():
        result = scheduler.poll()
        if result is None:
            await asyncio.sleep(0.01)  # No frames available
            continue

        camera_id, jpeg_bytes = result
        try:
            detection = inference_engine.run(jpeg_bytes, camera_id)

            # Decode frame for thumbnail (needed if violations found)
            frame_bgr = inference_engine.mm.preprocess_jpeg(jpeg_bytes)

            violations = alert_pipeline.process(detection, frame_bgr=frame_bgr)

            # Broadcast preview frame to dashboard
            preview_b64 = base64.b64encode(jpeg_bytes).decode()
            await ws_manager.broadcast_async(json.dumps({
                "type": "preview",
                "camera_id": camera_id,
                "frame_base64": preview_b64,
            }))

            if violations:
                types = [v.type for v in violations]
                print(f"[Main] ALERT: camera={camera_id} violations={types}")

        except Exception as e:
            print(f"[Main] Error processing frame from {camera_id}: {e}")


def run_edge_agent(config, local_bridge, stop_event):
    """Run edge agent in a separate thread for local cameras."""
    from edge import EdgeAgent

    try:
        agent = EdgeAgent(config, local_bridge=local_bridge)

        # Determine if any cameras need MQTT
        has_remote = any(
            not (str(c.get("source", "")).isdigit())
            for c in config.get("cameras", [])
        )
        if has_remote:
            agent.start_mqtt()

        agent.start_all_cameras()
        print(f"[Main] Edge agent started with {len(agent.source_manager.cameras)} cameras")

        # Keep alive
        while not stop_event.is_set():
            stop_event.wait(1)

        agent.stop()
    except Exception as e:
        print(f"[Main] Edge agent error: {e}")


async def main_async(config_path: str = "edge/config.yaml"):
    """Async main — wires everything together."""
    config = load_config(config_path)

    # Create shared infrastructure
    local_bridge = None
    local_camera_ids = []
    remote_camera_ids = []

    for cam in config.get("cameras", []):
        source = str(cam.get("source", ""))
        cid = cam.get("id", "")
        if source.isdigit():
            local_camera_ids.append(cid)
        else:
            remote_camera_ids.append(cid)

    stop_event = threading.Event()

    # Edge Agent (handles both local and remote cameras)
    if local_camera_ids or remote_camera_ids:
        from edge.local_bridge import LocalBridge
        local_bridge = LocalBridge()
        local_bridge._loop = asyncio.get_running_loop()
        edge_thread = threading.Thread(
            target=run_edge_agent,
            args=(config, local_bridge, stop_event),
            daemon=True,
        )
        edge_thread.start()
        await asyncio.sleep(0.5)  # Give edge time to start

    # Inference Engine
    from inference.model_manager import ModelManager
    from inference.detector import Detector
    from inference.scheduler import Scheduler

    mm = ModelManager()
    try:
        mm.load()
    except FileNotFoundError as e:
        print(f"[Main] Warning: Model not found ({e}), continuing without detection")
        # System still starts — dashboard and alerts work, just no detections

    detector = Detector(mm)
    scheduler = Scheduler()

    # Alert Pipeline
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

    # Register cameras with scheduler
    all_camera_ids = local_camera_ids + remote_camera_ids
    for cid in all_camera_ids:
        scheduler.register_camera(cid)

    # Start receivers
    from inference.local_receiver import LocalReceiver
    from inference.mqtt_subscriber import MQTTSubscriber

    # Local receiver
    local_receiver = None
    if local_camera_ids and local_bridge:
        local_receiver = LocalReceiver(local_bridge)
        await local_receiver.start(
            local_camera_ids,
            on_frame=lambda cid, jpeg: scheduler.add_frame(cid, jpeg),
        )
        print(f"[Main] Local receiver started for {local_camera_ids}")

    # MQTT subscriber (thread-safe callback -> scheduler)
    mqtt_sub = None
    if remote_camera_ids:
        mqtt_sub = MQTTSubscriber(
            broker=config.get("mqtt", {}).get("broker", "localhost"),
            port=config.get("mqtt", {}).get("port", 1883),
        )
        try:
            mqtt_sub.connect(
                on_frame=lambda cid, jpeg: scheduler.add_frame(cid, jpeg),
            )
            print(f"[Main] MQTT subscriber started for {remote_camera_ids}")
        except ConnectionError:
            print(f"[Main] Warning: Could not connect to MQTT broker, skipping remote cameras")
            mqtt_sub = None

    # Start dashboard
    dashboard_task = asyncio.create_task(run_dashboard(pipeline))

    # Run inference loop
    print("[Main] All subsystems started. Running...")
    try:
        await run_inference_loop(detector, pipeline, scheduler, stop_event)
    except asyncio.CancelledError:
        pass
    finally:
        print("[Main] Shutting down...")
        if local_receiver:
            await local_receiver.stop()
        if mqtt_sub:
            mqtt_sub.disconnect()
        stop_event.set()
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass


def main():
    """Entry point — handles signals and runs the async main."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "edge/config.yaml"

    loop = asyncio.new_event_loop()

    def sig_handler():
        print("\n[Main] Received SIGINT, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, sig_handler)
    loop.add_signal_handler(signal.SIGTERM, sig_handler)

    try:
        loop.run_until_complete(main_async(config_path))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        print("[Main] Goodbye.")


if __name__ == "__main__":
    main()

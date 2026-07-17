# Quadro T2000 4 GB benchmark plan

No realtime claim is valid until this plan runs on the physical edge device.

## Hardware inventory

Record the exact, unedited output of:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
nvidia-smi -q
lsusb -t
python3 scripts/discover_cameras.py --probe
```

Also record the ONNX Runtime version, active execution provider, model hashes,
camera codec, USB controller topology and thermal/power state.

## Workload matrix

Run one, two and three 1280×720 cameras at 15 capture FPS. For each camera
count, replay or stage 1, 3 and 5 simultaneous people. Compare detector inputs
512 and 640 with RTMPose-s at 256×192.

Measure these scheduling modes:

- detector batch 1, then batch 2/3 only if supported by the backend;
- normal tracks at 2–3 pose FPS;
- ROI-intersecting tracks at 5–8 pose FPS;
- `RAPID_TRANSITION`/`POSSIBLE_FALL` tracks at 10–15 pose FPS for a bounded
  candidate interval;
- ONNX Runtime CUDA FP32 first;
- TensorRT FP16 only after the ONNX baseline is stable.

Warm up for two minutes and measure each cell for at least 15 minutes. Run the
selected two-camera profile for 8–24 hours and repeat while the server is
offline, one camera is disconnected and a camera process is restarted.

## Required report

- capture, detector, pose and end-to-end analytics FPS per camera;
- fresh-frame and stale-drop counts;
- detector, pose, queue wait and event latency p50/p95/p99;
- peak VRAM/RAM and CUDA OOM count;
- GPU utilization, temperature and power throttling;
- active tracks and simultaneous pose tracks;
- camera reconnect duration and unaffected-peer FPS;
- event outbox throughput and disk growth during outage;
- replay precision/recall and false alarms by scenario.

## Initial acceptance targets

These are targets, not measured results:

- no CUDA OOM and no CPU provider fallback;
- at least 5 detector FPS per camera for two cameras;
- p95 analytics frame age below 500 ms;
- one camera disconnect causes no interruption to another camera;
- event processing overhead after the configured confirmation window below
  one second at p95;
- bounded memory, result queue and disk use during an eight-hour run.

Three-camera support is accepted only if it meets the same freshness and memory
criteria on the actual device. If it does not, reduce pose sampling, detector
input or analytics FPS through configuration and report the resulting accuracy
trade-off.

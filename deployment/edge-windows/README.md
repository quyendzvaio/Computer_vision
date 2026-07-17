# Windows edge deployment

The edge runtime runs natively on Windows so OpenCV can access USB cameras and
ONNX Runtime can use the NVIDIA CUDA provider without USB/WSL forwarding.

1. Install the NVIDIA driver and the CUDA/cuDNN versions required by the pinned
   `onnxruntime-gpu` wheel, then confirm `nvidia-smi` works.
2. Run `scripts/fetch_pretrained_models.py` and copy/export the verified models
   into `models/`. RTMDet must be exported by `scripts/export_rtmdet_onnx.py`.
3. Put a checksum-verified `WinSW-x64.exe` beside this file.
4. Configure numeric Windows camera sources in `edge_runtime/config/cameras.yaml`.
5. Enable only models whose runtime artifacts have passed
   `scripts/verify_pretrained_models.py`.
6. Run `install-service.ps1` from elevated PowerShell.

The service uses WinSW only as the Windows Service Control Manager adapter. The
application remains a normal Python process and handles graceful stop signals.

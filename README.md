### CSEG516-TERM-PROJECT
---
Subject : 

#### GPU Runtime Notes
- Make sure the GPU host has a compatible NVIDIA driver and `nvidia-container-toolkit` installed.
- Start containers with `docker run --gpus all ...` so the device is exposed; without it the image runs in CPU mode only.
- Check `nvidia-smi` to confirm the driver supports CUDA 12.5 before relying on GPU acceleration.

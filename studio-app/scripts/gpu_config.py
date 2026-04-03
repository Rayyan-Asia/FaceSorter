"""
GPU detection and ONNX Runtime provider configuration.

Call configure() once at startup — before loading InsightFace or any ONNX models.
Returns a dict describing the selected device and ONNX Runtime providers to use.
"""

import os
import sys
import subprocess


def _get_nvidia_info():
    """Query nvidia-smi for GPU name and VRAM. Returns (name, vram_gb) or (None, None)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                parts = lines[0].split(",")
                name = parts[0].strip()
                vram_gb = round(int(parts[1].strip()) / 1024, 1) if len(parts) > 1 else None
                return name, vram_gb
    except Exception:
        pass
    return None, None


def configure(logger=None):
    """
    Detect available compute device and return the best ONNX Runtime provider list.

    Provider priority:
      1. DmlExecutionProvider  — DirectX 12 GPU (Windows, any vendor, no CUDA toolkit needed)
      2. CUDAExecutionProvider — CUDA GPU (Linux / Windows with CUDA toolkit)
      3. CPUExecutionProvider  — fallback

    Returns:
        dict with keys:
            device      "GPU" | "CPU"
            providers   list[str]  — pass directly to insightface FaceAnalysis()
            gpu_name    str | None
            gpu_count   int
            vram_gb     float | None
    """
    info = {
        "device": "CPU",
        "providers": ["CPUExecutionProvider"],
        "gpu_name": None,
        "gpu_count": 0,
        "vram_gb": None,
    }

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        if "DmlExecutionProvider" in available:
            # DirectML — GPU via DirectX 12 (Windows native, no CUDA toolkit required)
            gpu_name, vram_gb = _get_nvidia_info()
            info.update({
                "device": "GPU",
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "gpu_name": gpu_name,
                "gpu_count": 1,
                "vram_gb": vram_gb,
            })
            msg = (
                "GPU (DirectML): "
                + (gpu_name or "unknown GPU")
                + (f" — {vram_gb} GB VRAM" if vram_gb else "")
            )

        elif "CUDAExecutionProvider" in available:
            # CUDA — best performance when CUDA toolkit is installed
            gpu_name, vram_gb = _get_nvidia_info()
            info.update({
                "device": "GPU",
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "gpu_name": gpu_name,
                "gpu_count": 1,
                "vram_gb": vram_gb,
            })
            msg = (
                "GPU (CUDA): "
                + (gpu_name or "unknown GPU")
                + (f" — {vram_gb} GB VRAM" if vram_gb else "")
            )

        else:
            gpu_name, _ = _get_nvidia_info()
            msg = "CPU"
            if gpu_name:
                msg += f" (system has {gpu_name} but no GPU provider available in onnxruntime)"

    except ImportError:
        msg = "onnxruntime not found — running on CPU"

    if logger:
        logger.info("Compute device: %s", msg)
    else:
        print(f"[device] {msg}", file=sys.stderr)

    return info

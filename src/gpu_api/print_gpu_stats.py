from __future__ import annotations

import subprocess
from typing import Any

import torch


def _query_nvidia_smi() -> dict[int, dict[str, float]]:
    """Return optional per-GPU utilization and memory values from nvidia-smi.

    Returns an empty dict when nvidia-smi is unavailable.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}

    stats: dict[int, dict[str, float]] = {}
    for row in out.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])
            mem_used_mb = float(parts[2])
            mem_total_mb = float(parts[3])
        except ValueError:
            continue
        stats[idx] = {
            "utilization_pct": util,
            "smi_used_gb": mem_used_mb / 1024.0,
            "smi_total_gb": mem_total_mb / 1024.0,
        }
    return stats


def get_gpu_stats() -> list[dict[str, Any]]:
    """Return per-GPU stats for footer/dashboard display.

    Each dict contains:
    - index
    - name
    - allocated_gb
    - reserved_gb
    - total_gb
    - free_gb
    - utilization_pct (if nvidia-smi available)
    - smi_used_gb / smi_total_gb (if nvidia-smi available)
    """
    if not torch.cuda.is_available():
        return []

    smi_stats = _query_nvidia_smi()
    per_gpu: list[dict[str, Any]] = []

    for idx in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(idx)
        total_gb = total_b / 1024**3
        free_gb = free_b / 1024**3
        allocated_gb = torch.cuda.memory_allocated(idx) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(idx) / 1024**3

        item: dict[str, Any] = {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "total_gb": total_gb,
            "free_gb": free_gb,
        }
        item.update(smi_stats.get(idx, {}))
        per_gpu.append(item)

    return per_gpu


def print_gpu_stats() -> tuple[float, float] | None:
    """Backward-compatible helper returning (allocated_gb, reserved_gb) for GPU0."""
    stats = get_gpu_stats()
    if not stats:
        return None
    gpu0 = stats[0]
    return float(gpu0["allocated_gb"]), float(gpu0["reserved_gb"])
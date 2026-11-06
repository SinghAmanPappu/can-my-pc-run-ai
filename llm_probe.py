#!/usr/bin/env python3
"""
llm_probe.py - Can My PC Run It? local AI hardware probe.

A single-file, standard-library-only Python script that probes your hardware,
measures memory/disk speed, estimates which local LLMs will run and how fast,
and prints a diagnosis + prescription.

No network access. Nothing installed. Read the file before running if you wish.
Run:  python llm_probe.py
Flags:
  --vram GB   Override detected GPU VRAM in GB (needed on non-NVIDIA/AMD GPUs).
  --ram  GB   Override total system RAM in GB.
  --json      Print machine-readable JSON instead of human text.
  --cleanup   Delete this probe file and exit (leaves nothing behind).

Estimates are rough (about +/- 40%) and derived from memory bandwidth.
Every speed is an estimate, not a guarantee.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Quantization levels: name -> approximate bits per weight
# ---------------------------------------------------------------------------
QUANTIZATIONS = [
    ("Q8_0", 8.5),
    ("Q6_K", 6.6),
    ("Q5_K_M", 5.7),
    ("Q4_K_M", 4.85),
    ("Q3_K_M", 3.9),
    ("Q2_K", 3.0),
]

# ---------------------------------------------------------------------------
# Model catalog: name -> {params_b, tag, moe_active_b or None, note}
# Default embedded list. If models.json exists next to this script, it is
# loaded instead so contributors can update models without touching code.
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    {"name": "Qwen 2.5 1.5B",   "params": 1.5,  "tag": "qwen2.5:1.5b",  "moe_active": None, "note": ""},
    {"name": "Llama 3.2 3B",    "params": 3.0,  "tag": "llama3.2:3b",   "moe_active": None, "note": ""},
    {"name": "Gemma 3 4B",      "params": 4.0,  "tag": "gemma3:4b",     "moe_active": None, "note": ""},
    {"name": "Llama 3.1 8B",    "params": 8.0,  "tag": "llama3.1:8b",   "moe_active": None, "note": ""},
    {"name": "Qwen3 8B",        "params": 8.0,  "tag": "qwen3:8b",      "moe_active": None, "note": ""},
    {"name": "Gemma 3 12B",     "params": 12.0, "tag": "gemma3:12b",    "moe_active": None, "note": ""},
    {"name": "Phi-4 14B",       "params": 14.0, "tag": "phi4:14b",      "moe_active": None, "note": ""},
    {"name": "gpt-oss 20B (MoE)", "params": 20.0, "tag": "gpt-oss:20b", "moe_active": 3.6,  "note": "Mixture-of-Experts"},
    {"name": "Qwen3 30B-A3B (MoE)", "params": 30.0, "tag": "qwen3:30b-a3b", "moe_active": 3.3, "note": "Mixture-of-Experts"},
    {"name": "Gemma 3 27B",     "params": 27.0, "tag": "gemma3:27b",    "moe_active": None, "note": ""},
    {"name": "Qwen3 32B",       "params": 32.0, "tag": "qwen3:32b",     "moe_active": None, "note": ""},
    {"name": "Llama 3.3 70B",   "params": 70.0, "tag": "llama3.3:70b",  "moe_active": None, "note": ""},
    {"name": "gpt-oss 120B (MoE)", "params": 120.0, "tag": "gpt-oss:120b", "moe_active": 5.1, "note": "Mixture-of-Experts"},
]

MODELS = list(DEFAULT_MODELS)  # may be replaced by models.json
CATALOG_SOURCE = "embedded defaults"

# ---------------------------------------------------------------------------
# GPU bandwidth assumptions (GB/s). Marketing figures, used for ranking only.
# ---------------------------------------------------------------------------
GPU_BANDWIDTH_CLASSES = {
    "entry": 200,
    "mid": 300,
    "upper-mid": 480,
    "high": 950,
}

# Specific card lookup (VRAM_GB -> bandwidth_GBs). Covers common cards.
GPU_CARD_TABLE = {
    # NVIDIA
    "gtx 1650": (4, 128),
    "gtx 1660": (6, 192),
    "gtx 1660 ti": (6, 288),
    "rtx 2060": (6, 336),
    "rtx 2060 super": (8, 448),
    "rtx 2070": (8, 448),
    "rtx 2070 super": (8, 448),
    "rtx 2080": (8, 448),
    "rtx 2080 super": (8, 448),
    "rtx 2080 ti": (11, 616),
    "rtx 3050": (8, 224),
    "rtx 3060": (12, 360),
    "rtx 3060 ti": (8, 448),
    "rtx 3070": (8, 448),
    "rtx 3070 ti": (8, 608),
    "rtx 3080": (10, 760),
    "rtx 3080 ti": (12, 912),
    "rtx 3090": (24, 936),
    "rtx 3090 ti": (24, 1008),
    "rtx 4060": (8, 272),
    "rtx 4060 ti": (8, 288),
    "rtx 4070": (12, 504),
    "rtx 4070 super": (12, 504),
    "rtx 4070 ti": (12, 608),
    "rtx 4070 ti super": (16, 672),
    "rtx 4080": (16, 717),
    "rtx 4080 super": (16, 736),
    "rtx 4090": (24, 1008),
    # AMD
    "radeon rx 6600": (8, 224),
    "radeon rx 6600 xt": (8, 256),
    "radeon rx 6700 xt": (12, 384),
    "radeon rx 6800": (16, 512),
    "radeon rx 6800 xt": (16, 512),
    "radeon rx 6900 xt": (16, 512),
    "radeon rx 6950 xt": (16, 512),
    "radeon rx 7600": (8, 288),
    "radeon rx 7700 xt": (12, 432),
    "radeon rx 7800 xt": (16, 624),
    "radeon rx 7900 xt": (20, 800),
    "radeon rx 7900 xtx": (24, 960),
    "radeon rx 7900 gre": (16, 576),
}

# ---------------------------------------------------------------------------
# RAM bandwidth assumptions (GB/s)
# ---------------------------------------------------------------------------
RAM_BANDWIDTH = {
    "ddr4_single": 20,
    "ddr4_dual": 40,
    "ddr5_dual": 70,
    "apple_base": 100,
    "apple_pro": 200,
    "apple_max": 400,
}

OS_RESERVE_GB = 2.5          # GB of RAM kept free for the system
GPU_VRAM_HEADROOM_GB = 0.5   # GB of VRAM kept free on the GPU
SHARED_MEMORY_USABLE = 0.85  # Apple Silicon usable share after reserve
MEM_COPY_MULTIPLIER = 1.3    # measured copy -> approx multi-threaded inference
EFFICIENCY_GPU = 0.6         # all-on-GPU speed factor
EFFICIENCY_SPLIT = 0.68      # CPU+GPU split speed factor (calibrated: real/est ~0.67)
EFFICIENCY_CPU = 0.6         # CPU-only speed factor

LAST_UPDATED = "2026-11-06"


def load_catalog():
    """Load models.json from the script directory if present; else use embedded defaults."""
    global MODELS, CATALOG_SOURCE, LAST_UPDATED
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        catalog_path = os.path.join(here, "models.json")
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("models"):
            normalized = []
            for m in data["models"]:
                normalized.append({
                    "name": m.get("name", ""),
                    "params": m.get("params", 0),
                    "tag": m.get("tag", ""),
                    "moe_active": m.get("moe_active", m.get("moeActive")),
                    "note": m.get("note", ""),
                })
            MODELS = normalized
            CATALOG_SOURCE = catalog_path
        if data.get("last_updated"):
            LAST_UPDATED = data["last_updated"]
    except Exception:
        # Keep embedded defaults — probe must remain single-file and work alone.
        pass


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def bytes_to_gb(b):
    return b / (1024 ** 3)


def detect_ram_gb():
    """Return total system RAM in GB."""
    try:
        import ctypes
        if platform.system() == "Windows":
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return bytes_to_gb(stat.ullTotalPhys)
    except Exception:
        pass
    # Fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return bytes_to_gb(kb * 1024)
    except Exception:
        pass
    return None


def detect_free_ram_gb():
    """Return free system RAM in GB (best effort)."""
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return bytes_to_gb(stat.ullAvailPhys)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem_free = mem_buff = mem_cached = None
            for line in f:
                if line.startswith("MemFree:"):
                    mem_free = int(line.split()[1]) * 1024
                elif line.startswith("Buffers:"):
                    mem_buff = int(line.split()[1]) * 1024
                elif line.startswith("Cached:"):
                    mem_cached = int(line.split()[1]) * 1024
            if mem_free is not None:
                avail = mem_free + (mem_buff or 0) + (mem_cached or 0)
                return bytes_to_gb(avail)
    except Exception:
        pass
    return None


def detect_cpu():
    """Return (cpu_name, threads)."""
    name = platform.processor() or "unknown CPU"
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                name = out
        except Exception:
            pass
    try:
        threads = os.cpu_count() or 1
    except Exception:
        threads = 1
    return name, threads


def detect_apple_silicon():
    """Return True if running on Apple Silicon (arm64 macOS)."""
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def run_cmd(cmd):
    """Run a command, return stdout text or None on failure."""
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15)
        return out
    except Exception:
        return None


def detect_nvidia_gpu():
    """Return (name, vram_gb) via nvidia-smi, or (None, None)."""
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None, None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None, None
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
    try:
        vram = float(parts[1]) / 1024.0  # MiB -> GiB approx
    except Exception:
        vram = None
    return name, vram


def detect_amd_gpu():
    """Return (name, vram_gb) via rocm-smi, or (None, None)."""
    out = run_cmd(["rocm-smi", "--productname", "--csv"])
    name = None
    if out:
        for line in out.strip().splitlines():
            if "GPU" in line and "," in line:
                parts = [p.strip() for p in line.split(",")]
                if parts and parts[0]:
                    name = parts[0]
                    break
    vram = None
    out2 = run_cmd(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if out2:
        for line in out2.strip().splitlines():
            m = re.search(r"(\d+)\s*MB", line)
            if m:
                vram = float(m.group(1)) / 1024.0
                break
    return name, vram


def detect_other_gpu():
    """Guess GPU name only (VRAM unknown).
    Windows: PowerShell CimInstance (wmic is removed on Windows 11 26200+).
    Linux: lspci.
    """
    name = None
    if platform.system() == "Windows":
        ps = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        ]
        out = run_cmd(ps)
        if out:
            for line in out.strip().splitlines():
                line = line.strip()
                if line and "microsoft basic" not in line.lower():
                    name = line
                    break
    else:
        out = run_cmd(["lspci", "-nn"])
        if out:
            for line in out.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    name = line
                    break
    return name, None


def detect_gpu(vram_override=None):
    """
    Detect GPU. Returns dict: {name, vram_gb, source, bandwidth_gbps, class}
    source: 'nvidia', 'amd', 'apple', 'other', 'unknown'
    """
    gpu = {"name": "unknown", "vram_gb": None, "source": "unknown",
           "bandwidth_gbps": None, "gpu_class": None}

    if detect_apple_silicon():
        chip = platform.processor() or "Apple Silicon"
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                chip = out
        except Exception:
            pass
        gpu["name"] = chip
        gpu["source"] = "apple"
        # Apple Silicon uses shared RAM; bandwidth handled separately
        gpu["vram_gb"] = None
        # Detect base / Pro / Max from chip name (e.g. "Apple M2 Pro")
        low = chip.lower()
        if "max" in low:
            gpu["bandwidth_gbps"] = RAM_BANDWIDTH["apple_max"]
            gpu["gpu_class"] = "high"
        elif "pro" in low:
            gpu["bandwidth_gbps"] = RAM_BANDWIDTH["apple_pro"]
            gpu["gpu_class"] = "high"
        else:
            gpu["bandwidth_gbps"] = RAM_BANDWIDTH["apple_base"]
            gpu["gpu_class"] = "mid"
        return gpu

    if vram_override is not None:
        gpu["vram_gb"] = vram_override

    name, vram = detect_nvidia_gpu()
    if name:
        gpu["name"] = name
        gpu["source"] = "nvidia"
        gpu["vram_gb"] = vram if vram is not None else gpu["vram_gb"]
        key = name.lower()
        for card, (card_vram, bw) in GPU_CARD_TABLE.items():
            if card in key:
                gpu["bandwidth_gbps"] = bw
                # derive class from bandwidth
                if bw >= 900:
                    gpu["gpu_class"] = "high"
                elif bw >= 450:
                    gpu["gpu_class"] = "upper-mid"
                elif bw >= 250:
                    gpu["gpu_class"] = "mid"
                else:
                    gpu["gpu_class"] = "entry"
                break
        if gpu["bandwidth_gbps"] is None:
            gpu["bandwidth_gbps"] = GPU_BANDWIDTH_CLASSES["mid"]
            gpu["gpu_class"] = "mid"
        return gpu

    name, vram = detect_amd_gpu()
    if name:
        gpu["name"] = name
        gpu["source"] = "amd"
        gpu["vram_gb"] = vram if vram is not None else gpu["vram_gb"]
        gpu["bandwidth_gbps"] = GPU_BANDWIDTH_CLASSES["mid"]
        gpu["gpu_class"] = "mid"
        return gpu

    name, vram = detect_other_gpu()
    if name:
        gpu["name"] = name
        gpu["source"] = "other"
        gpu["vram_gb"] = gpu["vram_gb"]  # may be None
        gpu["bandwidth_gbps"] = GPU_BANDWIDTH_CLASSES["entry"]
        gpu["gpu_class"] = "entry"
        return gpu

    return gpu


def classify_ram(ram_gb):
    """Guess RAM bandwidth class from total size (crude heuristic)."""
    if ram_gb is None:
        return RAM_BANDWIDTH["ddr4_single"], "ddr4_single"
    # Heuristic: 16GB+ on a modern machine is likely dual-channel DDR4/DDR5
    if ram_gb >= 16:
        return RAM_BANDWIDTH["ddr5_dual"], "ddr5_dual"
    if ram_gb >= 8:
        return RAM_BANDWIDTH["ddr4_dual"], "ddr4_dual"
    return RAM_BANDWIDTH["ddr4_single"], "ddr4_single"


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def measure_memory_copy_gbps(size_mb=256, runs=4):
    """Measure memory copy bandwidth (GB/s), best of N runs."""
    size_bytes = size_mb * 1024 * 1024
    best = float("inf")
    try:
        data = bytearray(size_bytes)
        dst = bytearray(size_bytes)
        for _ in range(runs):
            t0 = time.perf_counter()
            dst[:] = data[:]
            dt = time.perf_counter() - t0
            if dt > 0:
                best = min(best, dt)
    except Exception:
        pass
    if best == float("inf") or best <= 0:
        return None
    gb = size_bytes / (1024 ** 3)
    return gb / best


def measure_disk_speed_gbps(size_mb=128):
    """Measure disk write and read (GB/s). May be inflated by OS cache."""
    size_bytes = size_mb * 1024 * 1024
    results = {}
    tmp_path = os.path.join(tempfile_gettempdir(), "llm_probe_disk_test.bin")
    try:
        data = os.urandom(min(size_bytes, 4 * 1024 * 1024))
        # write
        t0 = time.perf_counter()
        with open(tmp_path, "wb") as f:
            written = 0
            while written < size_bytes:
                f.write(data[:min(len(data), size_bytes - written)])
                written += len(data[:min(len(data), size_bytes - written)])
        dt = time.perf_counter() - t0
        results["write_gbps"] = (size_bytes / (1024 ** 3)) / dt if dt > 0 else None
        # read
        t0 = time.perf_counter()
        with open(tmp_path, "rb") as f:
            f.read()
        dt = time.perf_counter() - t0
        results["read_gbps"] = (size_bytes / (1024 ** 3)) / dt if dt > 0 else None
    except Exception:
        results["write_gbps"] = None
        results["read_gbps"] = None
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    return results


def tempfile_gettempdir():
    import tempfile
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def model_memory_gb(params_b, bits):
    """Model memory (GB) = params_b * bits / 8 + 1 GB context/runtime."""
    return params_b * bits / 8.0 + 1.0


def bytes_per_word(params_b, bits, moe_active_b=None):
    """Bytes read per generated word (GB). MoE uses active params."""
    active = moe_active_b if moe_active_b else params_b
    return active * bits / 8.0  # GB (bytes expressed as billions = GB)


def speed_label(tps):
    if tps >= 20:
        return "fast"
    if tps >= 8:
        return "comfortable"
    if tps >= 3:
        return "slow but usable"
    return "very slow"


def estimate_for_model(model, ram_gb, vram_gb, ram_bw_gbps, gpu_bw_gbps,
                       gpu_class, apple_silicon, gpu_on=False):
    """
    Try quantizations highest->lowest. Return dict with best fit or None.
    """
    usable_ram = ram_gb - OS_RESERVE_GB
    if apple_silicon:
        usable_mem = usable_ram * SHARED_MEMORY_USABLE
    else:
        usable_mem = usable_ram

    gpu_mem = None
    if vram_gb is not None:
        gpu_mem = vram_gb - GPU_VRAM_HEADROOM_GB

    best = None
    for qname, bits in QUANTIZATIONS:
        mem = model_memory_gb(model["params"], bits)
        active = model["moe_active"] if model["moe_active"] else model["params"]
        b_per_word = active * bits / 8.0

        fits_gpu = gpu_mem is not None and mem <= gpu_mem
        fits_split = (gpu_mem is not None and mem <= (gpu_mem + usable_mem))
        fits_cpu = mem <= usable_mem

        if fits_gpu:
            tps = EFFICIENCY_GPU * gpu_bw_gbps / b_per_word
            mode = "all on GPU"
            f = 1.0
        elif fits_split:
            # choose f so GPU portion fills VRAM
            f = gpu_mem / mem if gpu_mem and mem else 0.0
            f = max(0.0, min(1.0, f))
            time_per_word = (f * b_per_word / gpu_bw_gbps) + ((1 - f) * b_per_word / ram_bw_gbps)
            tps = EFFICIENCY_SPLIT / time_per_word if time_per_word > 0 else 0
            mode = f"CPU+GPU split (~{f*100:.0f}% on GPU)"
        elif fits_cpu:
            tps = EFFICIENCY_CPU * ram_bw_gbps / b_per_word
            mode = "CPU only"
            f = 0.0
        else:
            continue  # doesn't fit at this quantization

        best = {
            "quant": qname,
            "bits": bits,
            "mem_gb": mem,
            "tps": tps,
            "mode": mode,
            "gpu_share": f if mode != "all on GPU" else 1.0,
            "fits_gpu": fits_gpu,
            "fits_cpu": fits_cpu,
        }
        break  # take first (highest quality) that fits

    if best is None:
        return None

    best["name"] = model["name"]
    best["tag"] = model["tag"]
    best["params"] = model["params"]
    best["moe_active"] = model["moe_active"]
    best["note"] = model["note"]
    best["speed_label"] = speed_label(best["tps"])
    best["tps_low"] = best["tps"] * 0.6
    best["tps_high"] = best["tps"] * 1.4
    return best


def pick_best(results):
    """Pick the best balanced pick: the most capable (largest) model that
    still runs at a comfortable speed (>= 8 toks/s), tie-broken by speed.
    If nothing reaches comfortable speed, fall back to the fastest."""
    viable = [r for r in results if r]
    if not viable:
        return None
    comfortable = [r for r in viable if r["tps"] >= 8]
    if comfortable:
        comfortable.sort(key=lambda r: (r["params"], r["tps"]), reverse=True)
        return comfortable[0]
    viable.sort(key=lambda r: (r["tps"], r["params"]), reverse=True)
    return viable[0]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def human_report(specs, results, best):
    lines = []
    lines.append("=" * 60)
    lines.append("CAN MY PC RUN IT? - hardware diagnosis")
    lines.append("=" * 60)
    lines.append("")

    lines.append("-- Senses --")
    lines.append(f"OS:           {specs['os']}")
    lines.append(f"CPU:          {specs['cpu_name']} ({specs['threads']} threads)")
    lines.append(f"RAM total:    {specs['ram_gb']:.1f} GB")
    lines.append(f"RAM free:     {specs['ram_free_gb']:.1f} GB" if specs['ram_free_gb'] else "RAM free:     unknown")
    lines.append(f"GPU:          {specs['gpu_name']}")
    lines.append(f"GPU source:   {specs['gpu_source']}")
    lines.append(f"VRAM:         {specs['vram_gb']:.1f} GB" if specs['vram_gb'] is not None else "VRAM:         unknown (pass --vram GB)")
    lines.append(f"Apple Silicon: {specs['apple_silicon']}")
    lines.append(f"RAM bandwidth: {specs['ram_bw_gbps']:.0f} GB/s ({specs['ram_bw_class']})")
    lines.append(f"GPU bandwidth: {specs['gpu_bw_gbps']:.0f} GB/s ({specs['gpu_class']})")
    lines.append("")

    lines.append("-- Pin-prick tests --")
    lines.append(f"Memory copy:   {specs['mem_copy_gbps']:.2f} GB/s (best of 4)" if specs['mem_copy_gbps'] else "Memory copy:   failed")
    ds = specs['disk']
    lines.append(f"Disk write:    {ds['write_gbps']:.2f} GB/s" if ds and ds['write_gbps'] else "Disk write:    failed")
    lines.append(f"Disk read:     {ds['read_gbps']:.2f} GB/s (may be inflated by OS cache)" if ds and ds['read_gbps'] else "Disk read:     failed")
    lines.append("")

    lines.append("-- Diagnosis (model fits & estimated speed) --")
    if not results:
        lines.append("No model fits with the detected specs.")
    else:
        lines.append(f"{'Model':<30} {'Q':<7} {'Mem(GB)':<9} {'Mode':<22} {'Est toks/s':<13} {'Label'}")
        lines.append("-" * 90)
        for r in results:
            if r:
                lines.append(f"{r['name']:<30} {r['quant']:<7} {r['mem_gb']:<9.1f} "
                             f"{r['mode']:<22} {r['tps']:<13.1f} ({r['tps_low']:.0f}-{r['tps_high']:.0f}) {r['speed_label']}")
            else:
                lines.append(f"{'':<30}")
    lines.append("")
    lines.append("Note: every speed above is an ESTIMATE (about +/- 40%). "
                 "Real speed depends on inference software, drivers, thermals, "
                 "background apps, and context length.")
    lines.append("")

    lines.append("-- Prescription --")
    if best:
        lines.append(f"Best balanced pick: {best['name']} ({best['tag']})")
        lines.append(f"  Quantization: {best['quant']}  ({best['bits']} bits/weight)")
        lines.append(f"  Memory needed: ~{best['mem_gb']:.1f} GB")
        lines.append(f"  Mode: {best['mode']}")
        lines.append(f"  Estimated speed: {best['tps']:.0f} tokens/sec ({best['tps_low']:.0f}-{best['tps_high']:.0f}) - {best['speed_label']}")
        lines.append(f"  Copy-paste: ollama run {best['tag']}")
        if best['note']:
            lines.append(f"  Note: {best['note']}")
    else:
        lines.append("Nothing fits. Try: close apps, add a second RAM stick (dual-channel), "
                     "use a 4-bit model, or try a smaller MoE model such as gpt-oss:20b.")
    lines.append("")

    lines.append("-- Tips --")
    lines.append("• Use 4-bit (Q4_K_M) models to fit more in memory.")
    lines.append("• Offload GPU layers: ollama run MODEL -ngl 99 (lower if you run out of memory).")
    lines.append("• Two matching RAM sticks (dual-channel) can nearly double CPU speed.")
    lines.append("• Plug in the charger and use performance mode on laptops.")
    lines.append("• MoE models (gpt-oss:20b, qwen3:30b-a3b) use far less active memory.")
    lines.append("")
    lines.append(f"Model catalog last updated: {LAST_UPDATED}  (source: {specs.get('catalog_source','embedded')})")
    lines.append("This site/tool is not affiliated with any model maker. No data is collected.")
    lines.append("=" * 60)
    return "\n".join(lines)


def json_report(specs, results, best):
    out = {
        "last_updated": LAST_UPDATED,
        "specs": specs,
        "results": results,
        "best": best,
        "disclaimer": "Speeds are estimates (+/- 40%). Not affiliated with any model maker. No data collected.",
    }
    return json.dumps(out, indent=2)


def cleanup_self():
    """Delete this file."""
    try:
        os.remove(os.path.abspath(__file__))
        print("Probe deleted itself. Nothing left behind.")
    except Exception as e:
        print(f"Could not delete self: {e}")


def main():
    parser = argparse.ArgumentParser(description="Can My PC Run It? - local AI hardware probe.")
    parser.add_argument("--vram", type=float, help="Override GPU VRAM in GB.")
    parser.add_argument("--ram", type=float, help="Override total RAM in GB.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--cleanup", action="store_true", help="Delete this probe file and exit.")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_self()
        return

    # Load external model catalog if present (falls back to embedded defaults)
    load_catalog()

    # ---- Detect hardware ----
    ram_gb = args.ram if args.ram else detect_ram_gb()
    ram_free_gb = detect_free_ram_gb()
    cpu_name, threads = detect_cpu()
    apple = detect_apple_silicon()
    gpu = detect_gpu(vram_override=args.vram)

    # RAM bandwidth
    ram_bw_gbps, ram_bw_class = classify_ram(ram_gb)
    if apple:
        # Use chip-specific bandwidth from GPU detection (base/Pro/Max)
        if gpu.get("bandwidth_gbps"):
            ram_bw_gbps = gpu["bandwidth_gbps"]
            if ram_bw_gbps >= 400:
                ram_bw_class = "apple_max"
            elif ram_bw_gbps >= 200:
                ram_bw_class = "apple_pro"
            else:
                ram_bw_class = "apple_base"
        else:
            ram_bw_gbps = RAM_BANDWIDTH["apple_base"]
            ram_bw_class = "apple_shared"

    # GPU bandwidth
    gpu_bw_gbps = gpu["bandwidth_gbps"] or GPU_BANDWIDTH_CLASSES["entry"]
    gpu_class = gpu["gpu_class"] or "entry"

    # ---- Benchmarks ----
    mem_copy_gbps = measure_memory_copy_gbps()
    disk = measure_disk_speed_gbps()

    # Calibrate bandwidth from measured memory copy if available
    if mem_copy_gbps:
        calibrated_ram_bw = mem_copy_gbps * MEM_COPY_MULTIPLIER
        # Use calibrated if it's in a sane range
        if 5 < calibrated_ram_bw < 2000:
            ram_bw_gbps = calibrated_ram_bw
            ram_bw_class = "measured"

    specs = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_name": cpu_name,
        "threads": threads,
        "ram_gb": ram_gb if ram_gb else 0,
        "ram_free_gb": ram_free_gb if ram_free_gb else 0,
        "ram_free_gb_raw": ram_free_gb,
        "gpu_name": gpu["name"],
        "gpu_source": gpu["source"],
        "vram_gb": gpu["vram_gb"],
        "apple_silicon": apple,
        "ram_bw_gbps": ram_bw_gbps,
        "ram_bw_class": ram_bw_class,
        "gpu_bw_gbps": gpu_bw_gbps,
        "gpu_class": gpu_class,
        "mem_copy_gbps": mem_copy_gbps,
        "disk": disk,
        "catalog_source": CATALOG_SOURCE,
    }

    # ---- Estimate all models ----
    results = []
    for model in MODELS:
        r = estimate_for_model(
            model,
            ram_gb=specs["ram_gb"],
            vram_gb=specs["vram_gb"],
            ram_bw_gbps=specs["ram_bw_gbps"],
            gpu_bw_gbps=specs["gpu_bw_gbps"],
            gpu_class=specs["gpu_class"],
            apple_silicon=apple,
        )
        results.append(r)

    best = pick_best(results)

    if args.json:
        print(json_report(specs, results, best))
    else:
        print(human_report(specs, results, best))


if __name__ == "__main__":
    main()

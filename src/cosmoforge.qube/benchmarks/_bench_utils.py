"""Shared helpers for benchmark scripts: provenance capture and result saving.

Metadata is collected purely from the Python runtime — no shell-out — so it
works on any cluster where the benchmark itself runs.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
REPO_ROOT = BENCHMARKS_DIR.parents[2]

_SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_NODELIST",
)
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
)
_DEPS_OF_INTEREST = (
    "cosmocore",
    "qube",
    "picslike",
    "harmlike",
    "numpy",
    "scipy",
    "healpy",
    "numba",
    "mpi4py",
    "pyyaml",
)


def _cpu_info() -> dict:
    info = {"count": os.cpu_count(), "model": "?"}
    try:
        with open("/proc/cpuinfo") as f:
            physical_ids: set[str] = set()
            for line in f:
                if line.startswith("model name") and info["model"] == "?":
                    info["model"] = line.split(":", 1)[1].strip()
                elif line.startswith("physical id"):
                    physical_ids.add(line.split(":", 1)[1].strip())
            if physical_ids:
                info["sockets"] = len(physical_ids)
    except OSError:
        info["model"] = platform.processor() or "?"
    return info


def _git_info() -> dict:
    info: dict = {"sha": None, "branch": None, "dirty": None}
    try:
        info["sha"] = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return info


def _blas_info() -> dict:
    out: dict = {}
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        for key in ("blas", "lapack"):
            entry = deps.get(key, {})
            if entry:
                out[key] = {
                    "name": entry.get("name", "?"),
                    "version": entry.get("version", "?"),
                }
    except (TypeError, AttributeError, KeyError):
        pass
    return out


def _packages() -> dict:
    versions: dict = {}
    for dist in distributions():
        name = dist.metadata.get("Name") if dist.metadata else None
        if name:
            versions[name] = dist.version
    return {k: versions[k] for k in _DEPS_OF_INTEREST if k in versions}


def _mpi_info() -> dict | None:
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return {
        "size": MPI.COMM_WORLD.Get_size(),
        "rank": MPI.COMM_WORLD.Get_rank(),
        "vendor": ".".join(str(x) for x in MPI.get_vendor()[:2])
        if hasattr(MPI, "get_vendor")
        else None,
    }


def capture_metadata(extra: dict | None = None) -> dict:
    md = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "uname": dict(platform.uname()._asdict()),
        "python": sys.version.split()[0],
        "cpu": _cpu_info(),
        "blas": _blas_info(),
        "packages": _packages(),
        "git": _git_info(),
        "thread_env": {k: os.environ.get(k) for k in _THREAD_ENV_KEYS},
        "slurm": {k: os.environ.get(k) for k in _SLURM_ENV_KEYS},
    }
    mpi = _mpi_info()
    if mpi is not None:
        md["mpi"] = mpi
    if extra:
        md.update(extra)
    return md


def save_results(name: str, results: dict, extra_metadata: dict | None = None) -> Path:
    """Save results to ``benchmarks/results/<name>_results.json`` with metadata."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{name}_results.json"
    payload = {
        "metadata": capture_metadata(extra_metadata),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path

"""ctypes loader for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYACVD_LIB") or os.path.join(ROOT, "dist", "libmojo-pyacvd.so")
I = ctypes.c_int64
F = ctypes.c_double
SIGNATURES = {
    "mpa_point_weights": ([I, I, I, I, I, I, I, I], None),
    "mpa_face_centroids": ([I, I, I, I], None),
    "mpa_face_normals": ([I, I, I, I], None),
    "mpa_farthest_seeds": ([I, I, I, I, I], None),
    "mpa_assign": ([I, I, I, I, I], F),
    "mpa_assign_range": ([I, I, I, I, I, I], None),
    "mpa_inertia": ([I, I, I, I], F),
    "mpa_weighted_update": ([I, I, I, I, I, I, I, I], F),
    "mpa_lloyd_gpu": ([I, I, I, I, I, I, I, I, I], I),
}

ASSIGN_BLOCK = 1 << 16
ASSIGN_CHUNK_MIN_WORK = 8_000_000
ASSIGN_WORKERS = 16


def assign(kernels: ctypes.CDLL, points, centers, labels) -> float:
    """Label every point with its nearest center.

    The assignment pass is compute bound: roughly eight flops per cluster per
    point against 32 bytes of point and label traffic, so arithmetic intensity
    is k/4 flops per byte and reaches two at eight clusters. Below the work
    threshold one call labels the whole array; above it the range entry point
    is fanned out over a thread pool, which releases the GIL for each foreign
    call.
    """
    n = len(points)
    k = len(centers)
    if n == 0 or k == 0:
        return 0.0
    if n * k < ASSIGN_CHUNK_MIN_WORK or n <= ASSIGN_BLOCK:
        return kernels.mpa_assign(
            addr(points), addr(centers), addr(labels), n, k
        )
    blocks = -(-n // ASSIGN_BLOCK)
    step = -(-blocks // ASSIGN_WORKERS)

    def run(first: int) -> None:
        kernels.mpa_assign_range(
            addr(points), addr(centers), addr(labels), k, first, min(first + step, n)
        )

    with ThreadPoolExecutor(max_workers=ASSIGN_WORKERS) as pool:
        for first in range(0, n, step):
            pool.submit(run, first)
    return kernels.mpa_inertia(addr(points), addr(centers), addr(labels), n)


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo is unavailable; run through `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run([mojo, "build", "--emit", "shared-lib", source, "-o", LIB], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (args, result) in SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes, fn.restype = args, result
    return _lib


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    return int(value.ctypes.data)

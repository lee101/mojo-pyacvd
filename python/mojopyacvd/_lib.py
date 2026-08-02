"""ctypes loader for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

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
    "mpa_weighted_update": ([I, I, I, I, I, I, I, I], F),
}


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

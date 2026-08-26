"""Benchmark mojo-pyacvd against upstream pyacvd on identical surface meshes."""

from __future__ import annotations

import math
import platform
import subprocess
import time

import pyacvd
import pyvista as pv

import mojopyacvd as mpa


def best(fn, repeat: int = 3) -> float:
    value = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def main() -> None:
    mesh = pv.Sphere(theta_resolution=180, phi_resolution=140).triangulate()
    print(f"Machine: {platform.platform()} | points={mesh.n_points:,}, triangles={mesh.n_cells:,}")
    gpu_ready = _gpu_has_headroom()
    if not gpu_ready:
        print("GPU benchmark skipped: less than 4000 MiB free or no NVIDIA GPU is available.")
    print("| kernel | mojo-pyacvd | pyacvd | upstream / Mojo | result |")
    print("|---|---:|---:|---:|---|")
    cases = [
        ("point_weights", lambda: mpa.point_weights(mesh, force_double=True), lambda: pyacvd.clustering.point_weights(mesh, force_double=True)),
        ("cluster 96 cells (20 iterations)", lambda: mpa.Clustering(mesh).cluster(96, maxiter=20), lambda: pyacvd.Clustering(mesh).cluster(96, maxiter=20)),
        ("full remesh 96 cells", lambda: _remesh(mpa, mesh), lambda: _remesh(pyacvd, mesh)),
    ]
    if gpu_ready:
        cases.insert(2, ("cluster GPU 96 cells (20 iterations)", lambda: mpa.Clustering(mesh).cluster(96, maxiter=20, device="gpu"), lambda: pyacvd.Clustering(mesh).cluster(96, maxiter=20)))
    for name, ours, theirs in cases:
        ours()
        a, b = best(ours), best(theirs)
        ratio = b / a
        result = "faster" if a < b else "slower"
        print(f"| {name} | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | {ratio:.2f}x | {result} |")


def _gpu_has_headroom() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        free_mib = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return bool(free_mib) and free_mib[0] >= 4000
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return False


def _remesh(package, mesh):
    clus = package.Clustering(mesh)
    clus.cluster(96, maxiter=20)
    return clus.create_mesh()


if __name__ == "__main__":
    main()

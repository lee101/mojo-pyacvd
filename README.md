# mojo-pyacvd

`mojo-pyacvd` is a standalone Mojo port of the compute-heavy pieces of
[pyacvd](https://github.com/pyvista/pyacvd): area-weight calculation, triangle
geometry, area-weighted Voronoi assignment, centroid updates, and triangular
dual-mesh reconstruction.  It provides a PyVista-facing `Clustering` class with
the same covered method signatures as upstream.

The covered API is `Clustering`, `cluster`, `fast_cluster`, `subdivide`,
`create_mesh`, `cluster_centroid`, `cluster_norm`, `point_weights`,
`face_centroid_arrays`, `face_normals_array`, and `polydata_from_faces`. Each
covered public operation has a parity or contract test in `tests/`.
The topology-constrained ACVD optimizer, ray-cast centre projection, neighbor
utilities, and PyVista `mesh.acvd` accessor are not yet ported.  The solver is
deterministic area-weighted Lloyd/Voronoi clustering; it yields valid uniform
triangular dual meshes but does not promise identical cluster labels to
upstream's topology-aware optimizer.

## Install

```bash
pixi install
pixi run build
```

Everything runs through Pixi:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import pyvista as pv
import mojopyacvd as pyacvd

mesh = pv.Sphere(theta_resolution=32, phi_resolution=24).triangulate()
clus = pyacvd.Clustering(mesh)
clus.cluster(100)
remesh = clus.create_mesh()
assert remesh.n_points == 100
```

## How it works

The Python layer keeps PyVista objects and topology handling where that API is
useful. It validates finite float32/float64 point data and integer triangle
indices, then passes C-contiguous float64 points and int64 connectivity to one
Mojo shared library through `ctypes`; buffers cross the ABI as integer addresses
and remain owned by NumPy for the entire native call. Mojo computes triangle areas,
face geometry, nearest Voronoi cells, and weighted centroids in flat row-major
memory, avoiding temporary Python arrays in the iterative hot path.

## Benchmarks

Measured with `pixi run bench` on this machine. Times are the best of three;
the full-remesh comparison includes Python/PyVista reconstruction. The mesh had
24,842 points and 49,680 triangles.

| kernel | mojo-pyacvd | pyacvd | upstream / Mojo | result |
|---|---:|---:|---:|---|
| point_weights | 1.63 ms | 1.17 ms | 0.72x | slower |
| cluster 96 cells (20 iterations) | 116.03 ms | 11.22 ms | 0.10x | slower |
| full remesh 96 cells | 144.48 ms | 103.22 ms | 0.71x | slower |

Upstream's compiled, topology-aware ACVD implementation wins these runs.  This
port deliberately reports that result rather than claiming an unmeasured
speedup; its value is a small, standalone Mojo kernel boundary and a
deterministic weighted-Voronoi implementation.

There is no GPU path. The current kernels are CPU-only.

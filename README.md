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

CPU execution is the default. `clus.cluster(100, device="gpu")` explicitly
requests the optional GPU assignment path; it silently falls back to CPU when a
GPU is unavailable, has less than 4000 MiB free, or cannot allocate its buffers.

## How it works

The Python layer keeps PyVista objects and topology handling where that API is
useful. It validates finite float32/float64 point data and integer triangle
indices, then passes C-contiguous float64 points and int64 connectivity to one
Mojo shared library through `ctypes`; buffers cross the ABI as integer addresses
and remain owned by NumPy for the entire native call. Mojo computes triangle areas,
face geometry, nearest Voronoi cells, and weighted centroids in flat row-major
memory, avoiding temporary Python arrays in the iterative hot path.

Nearest-centre assignment is SIMD-vectorized across points, with a scalar tail,
and splits into contiguous blocks above 8 million point-centre comparisons.
That pass is compute bound rather than bandwidth bound -- about eight flops per
cluster per point against 32 bytes of point and label traffic, so arithmetic
intensity is k/4 flops per byte and reaches two at eight clusters -- so above
the threshold the blocks are fanned out over a `ThreadPoolExecutor` that
releases the GIL for each foreign call. The GPU
path keeps points resident while Lloyd iterations run and copies only centres
and labels per iteration. The benchmark mesh uses less than 1 MiB of device
storage; the runtime rejects paths requiring 2 GiB or more.

## Benchmarks

Measured with `pixi run bench` on this machine. Times are the best of three;
the full-remesh comparison includes Python/PyVista reconstruction. The mesh had
24,842 points and 49,680 triangles.

| kernel | mojo-pyacvd | pyacvd | upstream / Mojo | result |
|---|---:|---:|---:|---|
| point_weights | 0.97 ms | 0.69 ms | 0.71x | slower |
| cluster 96 cells (20 iterations) | 43.22 ms | 10.08 ms | 0.23x | slower |
| cluster GPU 96 cells (20 iterations) | 15.46 ms | 8.83 ms | 0.57x | slower |
| full remesh 96 cells | 64.77 ms | 86.22 ms | 1.33x | faster |

Upstream still wins the standalone point-weight and clustering comparisons;
the Mojo port wins the full-remesh measurement. The point-weight face scatter
is low arithmetic intensity and remains CPU-only: its native kernel measured
0.41 ms, while validation, float64 conversion, and output setup dominate the
public call. Moving that work to the GPU would add transfers without enough
computation to recover their cost.

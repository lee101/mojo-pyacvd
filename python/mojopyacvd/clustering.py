"""PyVista-compatible area-weighted clustered Voronoi remeshing."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from ._lib import addr, assign, f64, i64, lib


def _points(value, *, name: str = "points") -> np.ndarray:
    """Validate a public point array before passing its address to Mojo."""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected {name} shaped (n, 3).")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have float32 or float64 dtype.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return f64(array)


def _triangles(value, npoints: int) -> np.ndarray:
    """Validate triangle connectivity before it becomes an unchecked pointer."""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Expected triangular faces shaped (n, 3).")
    if array.dtype.kind not in "iu":
        raise TypeError("faces must have an integer dtype.")
    if array.size:
        if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
            raise ValueError("faces contain indices outside int64 range.")
        if np.any(array < 0) or np.any(array >= npoints):
            raise ValueError("faces contain an index outside the points array.")
    return i64(array)


def _faces(mesh: pv.PolyData) -> np.ndarray:
    if not mesh.is_all_triangles:
        raise ValueError("Input mesh must be composed of all triangles. Hint: `mesh.triangulate` first.")
    return _triangles(mesh._connectivity_array.reshape(-1, 3), mesh.n_points)


def polydata_from_faces(points, faces) -> pv.PolyData:
    points = _points(points)
    faces = _triangles(faces, len(points))
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    return pv.PolyData(points, packed.ravel())


def point_weights(mesh: pv.PolyData, additional_weights=None, num_threads: int = 2, force_double: bool = False):
    """Return ACVD point areas and area-weighted point coordinates."""
    del num_threads
    points, faces = _points(mesh.points), _faces(mesh)
    if additional_weights is None:
        extra = np.empty(0, dtype=np.float64)
    else:
        raw_extra = np.asarray(additional_weights)
        if raw_extra.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("additional_weights must have float32 or float64 dtype.")
        extra = f64(raw_extra).reshape(-1)
    if len(extra) not in (0, len(points)):
        raise ValueError("additional_weights must have one value per point.")
    if not np.all(np.isfinite(extra)):
        raise ValueError("additional_weights must contain only finite values.")
    area = np.empty(len(points), dtype=np.float64)
    wcent = np.empty_like(points)
    lib().mpa_point_weights(addr(points), addr(faces), addr(extra), addr(area), addr(wcent), len(points), len(faces), int(len(extra) != 0))
    if not force_double and mesh.points.dtype == np.float32:
        return area.astype(np.float32), wcent.astype(np.float32)
    return area, wcent


def face_centroid_arrays(points, faces):
    dtype = np.asarray(points).dtype
    points = _points(points)
    faces = _triangles(faces, len(points))
    dst = np.empty((len(faces), 3), dtype=np.float64)
    lib().mpa_face_centroids(addr(points), addr(faces), addr(dst), len(faces))
    return dst.astype(dtype, copy=False)


def face_normals_array(points, faces):
    dtype = np.asarray(points).dtype
    points = _points(points)
    faces = _triangles(faces, len(points))
    dst = np.empty((len(faces), 3), dtype=np.float64)
    lib().mpa_face_normals(addr(points), addr(faces), addr(dst), len(faces))
    return dst.astype(dtype, copy=False)


def _cluster_centroid(points, area, clusters):
    points, area, clusters = f64(points), f64(area).reshape(-1), i64(clusters).reshape(-1)
    valid = clusters >= 0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    nclus = int(clusters[valid].max()) + 1
    mass = np.bincount(clusters[valid], weights=area[valid], minlength=nclus)
    mass[mass == 0] = 1.0
    return np.vstack([np.bincount(clusters[valid], weights=area[valid] * points[valid, j], minlength=nclus) / mass for j in range(3)]).T


def _farthest_seeds(points: np.ndarray, nclus: int) -> np.ndarray:
    seeds = np.empty((nclus, 3), dtype=np.float64)
    nearest = np.empty(len(points), dtype=np.float64)
    lib().mpa_farthest_seeds(addr(points), addr(seeds), addr(nearest), len(points), nclus)
    return seeds


class Clustering:
    """Uniform point clustering based on area-weighted Voronoi cells.

    The public methods and result attributes follow :class:`pyacvd.Clustering`.
    Unlike upstream's topology-aware solver, this implementation uses a
    deterministic weighted Lloyd solver, then reconstructs the dual mesh.
    """

    def __init__(self, mesh: pv.PolyData, weights=None) -> None:
        self.mesh = mesh.copy(deep=True)
        _faces(self.mesh)
        _points(self.mesh.points)
        self.clusters: np.ndarray | None = None
        self.nclus: int | None = None
        self.area, self.wcent = point_weights(self.mesh, weights, force_double=True)
        self.area[self.area == 0] = 1e-10

    def _update_data(self, weights=None) -> None:
        self.area, self.wcent = point_weights(self.mesh, weights, force_double=True)
        self.area[self.area == 0] = 1e-10

    def cluster(self, nclus: int, maxiter: int = 100, debug: bool = False, iso_try: int = 10, init_only: bool = False, device: str = "cpu") -> np.ndarray:
        del debug, iso_try
        if nclus < 1:
            raise ValueError("nclus must be positive.")
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'.")
        points = f64(self.mesh.points)
        self.nclus = min(int(nclus), len(points))
        if nclus >= len(points):
            self.clusters = np.arange(len(points), dtype=np.int32)
            self._centers = points.copy()
            return self.clusters
        centers = _farthest_seeds(points, self.nclus)
        labels = np.empty(len(points), dtype=np.int64)
        sums = np.empty_like(centers)
        masses = np.empty(self.nclus, dtype=np.float64)
        kernels = lib()
        iterations = 1 if init_only else maxiter
        used_gpu = False
        if device == "gpu":
            initial_centers = centers.copy()
            used_gpu = bool(kernels.mpa_lloyd_gpu(addr(points), addr(self.area), addr(labels), addr(centers), addr(sums), addr(masses), len(points), self.nclus, iterations))
            if not used_gpu:
                centers[:] = initial_centers
        if not used_gpu:
            for _ in range(iterations):
                assign(kernels, points, centers, labels)
                shift = kernels.mpa_weighted_update(addr(points), addr(self.area), addr(labels), addr(centers), addr(sums), addr(masses), len(points), self.nclus)
                if shift <= 1e-20:
                    break
            assign(kernels, points, centers, labels)
        self.clusters = labels.astype(np.int32)
        self._centers = centers
        return self.clusters

    def fast_cluster(self, nclus: int) -> np.ndarray:
        return self.cluster(nclus, maxiter=1, init_only=True)

    def subdivide(self, nsub: int) -> None:
        if nsub < 0:
            raise ValueError("nsub must be non-negative.")
        self.mesh = self.mesh.subdivide(nsub, subfilter="linear").clean()
        self._update_data()

    @property
    def cluster_centroid(self) -> np.ndarray:
        if self.clusters is None:
            raise RuntimeError("Clusters have not been initialized.")
        return _cluster_centroid(self.mesh.points, self.area, self.clusters)

    @property
    def cluster_norm(self) -> np.ndarray:
        if self.clusters is None or self.nclus is None:
            raise RuntimeError("Clusters have not been initialized.")
        normals = self.mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False).point_normals
        accum = np.vstack([np.bincount(self.clusters, weights=normals[:, j] * self.area, minlength=self.nclus) for j in range(3)]).T
        length = np.linalg.norm(accum, axis=1)
        length[length == 0] = 1.0
        return accum / length[:, None]

    def create_mesh(self, moveclus: bool = True, flipnorm: bool = True, clean: bool = True) -> pv.PolyData:
        if self.clusters is None or self.nclus is None:
            raise RuntimeError("Clusters have not been initialized.")
        del moveclus
        faces = _faces(self.mesh)
        clustered_faces = np.sort(self.clusters[faces], axis=1)
        clustered_faces = clustered_faces[np.all(np.diff(clustered_faces, axis=1) != 0, axis=1)]
        if len(clustered_faces):
            clustered_faces = np.unique(clustered_faces, axis=0)
        points = self.cluster_centroid
        result = polydata_from_faces(points, clustered_faces)
        if flipnorm and result.n_cells:
            normals = face_normals_array(points, clustered_faces)
            mean = self.cluster_norm[clustered_faces].sum(axis=1)
            reverse = np.einsum("ij,ij->i", normals, mean) < 0
            if np.any(reverse):
                clustered_faces[reverse] = clustered_faces[reverse, ::-1]
                result = polydata_from_faces(points, clustered_faces)
        self.remesh = result.clean() if clean else result
        return self.remesh

    def plot(self, random_color: bool = True, **kwargs):
        if self.clusters is None or self.nclus is None:
            return self.mesh.plot(notebook=False, **kwargs)
        rng = np.random.default_rng(0)
        palette = rng.random(self.nclus) if random_color else np.linspace(0, 1, self.nclus)
        return self.mesh.plot(notebook=False, scalars=palette[self.clusters], **kwargs)

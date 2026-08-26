"""Numerical parity checks against the installed upstream pyacvd package."""

from __future__ import annotations

import numpy as np
import pytest
import pyacvd
import pyvista as pv

import mojopyacvd as mpa
from mojopyacvd.clustering import _cluster_centroid, _farthest_seeds


@pytest.fixture(scope="module")
def mesh():
    return pv.Sphere(radius=1.0, theta_resolution=18, phi_resolution=14).triangulate()


def test_point_weights_matches_upstream(mesh):
    ours = mpa.point_weights(mesh, force_double=True)
    theirs = pyacvd.clustering.point_weights(mesh, force_double=True)
    assert np.allclose(ours[0], theirs[0], rtol=1e-13, atol=1e-14)
    assert np.allclose(ours[1], theirs[1], rtol=1e-13, atol=1e-14)


def test_weighted_point_weights_matches_upstream(mesh):
    weights = np.linspace(0.5, 2.0, mesh.n_points)
    ours = mpa.point_weights(mesh, weights, force_double=True)
    theirs = pyacvd.clustering.point_weights(mesh, weights, force_double=True)
    assert np.allclose(ours[0], theirs[0], rtol=1e-13, atol=1e-14)
    assert np.allclose(ours[1], theirs[1], rtol=1e-13, atol=1e-14)


def test_face_centroids_and_normals_match_upstream(mesh):
    faces = mesh._connectivity_array.reshape(-1, 3)
    assert np.allclose(
        mpa.face_centroid_arrays(mesh.points, faces),
        pyacvd.clustering.face_centroid_arrays(mesh.points, faces),
    )
    assert np.allclose(
        mpa.face_normals_array(mesh.points, faces),
        pyacvd.clustering.face_normals_array(mesh.points, faces),
    )


def test_cluster_centroids_match_upstream(mesh):
    labels = np.arange(mesh.n_points, dtype=np.int32) % 11
    area, _ = mpa.point_weights(mesh, force_double=True)
    ours = _cluster_centroid(mesh.points, area, labels)
    theirs = pyacvd.clustering._cluster_centroid(mesh.points, area, labels)
    assert np.allclose(ours, theirs)


def test_farthest_seeds_simd_tail_matches_reference():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25], [-1.0, 0.5, 1.0], [0.5, -1.0, 0.0],
         [2.0, 1.0, -0.5], [-0.5, -1.5, 0.5], [0.25, 0.75, 1.5]], dtype=np.float64,
    )
    seeds = _farthest_seeds(points, 4)
    reference = np.empty_like(seeds)
    reference[0] = points[0]
    nearest = np.sum((points - reference[0]) ** 2, axis=1)
    for i in range(1, len(reference)):
        reference[i] = points[np.argmax(nearest)]
        nearest = np.minimum(nearest, np.sum((points - reference[i]) ** 2, axis=1))
    assert np.array_equal(seeds, reference)


def test_assign_parallel_threshold_matches_serial():
    rng = np.random.default_rng(42)
    centers = rng.random((257, 3))
    small = rng.random((257, 3))
    large = np.tile(small, (256, 1))
    small_labels = np.empty(len(small), dtype=np.int64)
    large_labels = np.empty(len(large), dtype=np.int64)
    kernels = mpa._lib.lib()
    kernels.mpa_assign(mpa._lib.addr(small), mpa._lib.addr(centers), mpa._lib.addr(small_labels), len(small), len(centers))
    kernels.mpa_assign(mpa._lib.addr(large), mpa._lib.addr(centers), mpa._lib.addr(large_labels), len(large), len(centers))
    assert np.all(large_labels.reshape(-1, len(small)) == small_labels)


def test_gpu_cluster_matches_cpu_or_falls_back(mesh):
    cpu = mpa.Clustering(mesh)
    gpu = mpa.Clustering(mesh)
    assert np.array_equal(cpu.cluster(24, maxiter=5), gpu.cluster(24, maxiter=5, device="gpu"))
    with pytest.raises(ValueError):
        gpu.cluster(24, device="cuda")


def test_subdivide_matches_upstream_topology(mesh):
    ours, theirs = mpa.Clustering(mesh), pyacvd.Clustering(mesh)
    ours.subdivide(1)
    theirs.subdivide(1)
    assert ours.mesh.n_points == theirs.mesh.n_points
    assert ours.mesh.n_cells == theirs.mesh.n_cells
    assert np.allclose(ours.mesh.points, theirs.mesh.points)


def test_cluster_api_and_mesh_contract(mesh):
    ours, theirs = mpa.Clustering(mesh), pyacvd.Clustering(mesh)
    labels = ours.cluster(24, maxiter=30)
    reference_labels = theirs.cluster(24, maxiter=30)
    assert labels.dtype == np.int32
    assert labels.shape == reference_labels.shape == (mesh.n_points,)
    assert labels.min() >= 0 and labels.max() < 24
    assert ours.nclus == theirs.nclus == 24
    result = ours.create_mesh(moveclus=True, flipnorm=True, clean=True)
    reference = theirs.create_mesh(moveclus=True, flipnorm=True, clean=True)
    assert result.is_all_triangles
    assert result.n_points == reference.n_points == 24
    assert result.n_cells > 0
    assert np.all(np.isfinite(result.points))
    assert np.all(result.faces.reshape(-1, 4)[:, 0] == 3)


def test_cluster_count_is_clamped_like_upstream(mesh):
    ours, theirs = mpa.Clustering(mesh), pyacvd.Clustering(mesh)
    assert np.array_equal(ours.cluster(mesh.n_points + 5), theirs.cluster(mesh.n_points + 5))
    assert ours.nclus == theirs.nclus == mesh.n_points


def test_triangle_requirement_matches_upstream():
    quad = pv.Plane(i_resolution=1, j_resolution=1)
    with pytest.raises(ValueError):
        mpa.Clustering(quad)
    with pytest.raises(ValueError):
        pyacvd.Clustering(quad)


def test_uninitialized_create_mesh_matches_upstream(mesh):
    with pytest.raises(RuntimeError):
        mpa.Clustering(mesh).create_mesh()
    with pytest.raises(RuntimeError):
        pyacvd.Clustering(mesh).create_mesh()


def test_public_helpers_reject_unsafe_ffi_inputs():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(TypeError):
        mpa.face_centroid_arrays(points.astype(np.int64), faces)
    with pytest.raises(TypeError):
        mpa.face_normals_array(points, faces.astype(float))
    with pytest.raises(ValueError):
        mpa.face_centroid_arrays(points, np.array([[0, 1, 3]], dtype=np.int64))
    with pytest.raises(ValueError):
        mpa.polydata_from_faces(points, np.array([[0, -1, 2]], dtype=np.int64))


def test_remaining_covered_api_smoke(mesh):
    clus = mpa.Clustering(mesh)
    labels = clus.fast_cluster(12)
    assert labels.shape == (mesh.n_points,)
    assert clus.cluster_norm.shape == (12, 3)
    result = mpa.polydata_from_faces(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    assert result.n_points == 3 and result.n_cells == 1

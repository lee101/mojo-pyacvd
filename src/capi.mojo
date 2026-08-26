"""Numerical kernels for area-weighted clustered Voronoi remeshing.

The ABI owns no memory.  Python passes contiguous float64/int64 buffers as
addresses and retains every allocation for the duration of each call.
"""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime ASSIGN_BLOCK_SIZE = 1024


def triangle_area(points: FPtr, a: Int, b: Int, c: Int) -> Float64:
    var ax = points[3 * a]
    var ay = points[3 * a + 1]
    var az = points[3 * a + 2]
    var abx = points[3 * b] - ax
    var aby = points[3 * b + 1] - ay
    var abz = points[3 * b + 2] - az
    var acx = points[3 * c] - ax
    var acy = points[3 * c + 1] - ay
    var acz = points[3 * c + 2] - az
    var cx = aby * acz - abz * acy
    var cy = abz * acx - abx * acz
    var cz = abx * acy - aby * acx
    return 0.5 * sqrt(cx * cx + cy * cy + cz * cz)


def squared_distances_to_point(points: FPtr, dst: FPtr, n: Int, x: Float64, y: Float64, z: Float64):
    var i = 0
    while i + W <= n:
        var dx = (points + 3 * i).strided_load[width=W](3) - x
        var dy = (points + 3 * i + 1).strided_load[width=W](3) - y
        var dz = (points + 3 * i + 2).strided_load[width=W](3) - z
        dst.store(i, dx * dx + dy * dy + dz * dz)
        i += W
    while i < n:
        var dx = points[3 * i] - x
        var dy = points[3 * i + 1] - y
        var dz = points[3 * i + 2] - z
        dst[i] = dx * dx + dy * dy + dz * dz
        i += 1


@export("mpa_farthest_seeds")
def mpa_farthest_seeds(points_addr: Int, seeds_addr: Int, nearest_addr: Int, n: Int, k: Int) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var seeds = FPtr(unsafe_from_address=seeds_addr)
    var nearest = FPtr(unsafe_from_address=nearest_addr)
    seeds[0] = points[0]
    seeds[1] = points[1]
    seeds[2] = points[2]
    squared_distances_to_point(points, nearest, n, seeds[0], seeds[1], seeds[2])
    for cluster in range(1, k):
        var index = 0
        var largest = nearest[0]
        for i in range(1, n):
            if nearest[i] > largest:
                largest = nearest[i]
                index = i
        var offset = 3 * cluster
        var point_offset = 3 * index
        seeds[offset] = points[point_offset]
        seeds[offset + 1] = points[point_offset + 1]
        seeds[offset + 2] = points[point_offset + 2]
        var i = 0
        while i + W <= n:
            var dx = (points + 3 * i).strided_load[width=W](3) - seeds[offset]
            var dy = (points + 3 * i + 1).strided_load[width=W](3) - seeds[offset + 1]
            var dz = (points + 3 * i + 2).strided_load[width=W](3) - seeds[offset + 2]
            var dist = dx * dx + dy * dy + dz * dz
            nearest.store(i, min(nearest.load[width=W](i), dist))
            i += W
        while i < n:
            var dx = points[3 * i] - seeds[offset]
            var dy = points[3 * i + 1] - seeds[offset + 1]
            var dz = points[3 * i + 2] - seeds[offset + 2]
            var dist = dx * dx + dy * dy + dz * dz
            if dist < nearest[i]:
                nearest[i] = dist
            i += 1


@export("mpa_point_weights")
def mpa_point_weights(
    points_addr: Int, faces_addr: Int, extra_addr: Int, area_addr: Int,
    wcent_addr: Int, npoints: Int, nfaces: Int, has_extra: Int
) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var faces = IPtr(unsafe_from_address=faces_addr)
    var area = FPtr(unsafe_from_address=area_addr)
    var wcent = FPtr(unsafe_from_address=wcent_addr)
    var i = 0
    while i + W <= npoints:
        area.store(i, SIMD[DType.float64, W](0.0))
        i += W
    while i < npoints:
        area[i] = 0.0
        i += 1
    for t in range(nfaces):
        var a = faces[3 * t]
        var b = faces[3 * t + 1]
        var c = faces[3 * t + 2]
        var value = triangle_area(points, a, b, c)
        area[a] += value
        area[b] += value
        area[c] += value
    if has_extra != 0:
        var extra = FPtr(unsafe_from_address=extra_addr)
        i = 0
        while i + W <= npoints:
            var weight = area.load[width=W](i) * extra.load[width=W](i)
            (wcent + 3 * i).strided_store((points + 3 * i).strided_load[width=W](3) * weight, 3)
            (wcent + 3 * i + 1).strided_store((points + 3 * i + 1).strided_load[width=W](3) * weight, 3)
            (wcent + 3 * i + 2).strided_store((points + 3 * i + 2).strided_load[width=W](3) * weight, 3)
            i += W
        while i < npoints:
            var weight = area[i] * extra[i]
            wcent[3 * i] = points[3 * i] * weight
            wcent[3 * i + 1] = points[3 * i + 1] * weight
            wcent[3 * i + 2] = points[3 * i + 2] * weight
            i += 1
    else:
        i = 0
        while i + W <= npoints:
            var weight = area.load[width=W](i)
            (wcent + 3 * i).strided_store((points + 3 * i).strided_load[width=W](3) * weight, 3)
            (wcent + 3 * i + 1).strided_store((points + 3 * i + 1).strided_load[width=W](3) * weight, 3)
            (wcent + 3 * i + 2).strided_store((points + 3 * i + 2).strided_load[width=W](3) * weight, 3)
            i += W
        while i < npoints:
            wcent[3 * i] = points[3 * i] * area[i]
            wcent[3 * i + 1] = points[3 * i + 1] * area[i]
            wcent[3 * i + 2] = points[3 * i + 2] * area[i]
            i += 1


@export("mpa_face_centroids")
def mpa_face_centroids(points_addr: Int, faces_addr: Int, dst_addr: Int, nfaces: Int) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var faces = IPtr(unsafe_from_address=faces_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for t in range(nfaces):
        var a = faces[3 * t]
        var b = faces[3 * t + 1]
        var c = faces[3 * t + 2]
        for j in range(3):
            dst[3 * t + j] = (points[3 * a + j] + points[3 * b + j] + points[3 * c + j]) / 3.0


@export("mpa_face_normals")
def mpa_face_normals(points_addr: Int, faces_addr: Int, dst_addr: Int, nfaces: Int) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var faces = IPtr(unsafe_from_address=faces_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for t in range(nfaces):
        var a = faces[3 * t]
        var b = faces[3 * t + 1]
        var c = faces[3 * t + 2]
        var ax = points[3 * a]
        var ay = points[3 * a + 1]
        var az = points[3 * a + 2]
        var abx = points[3 * b] - ax
        var aby = points[3 * b + 1] - ay
        var abz = points[3 * b + 2] - az
        var acx = points[3 * c] - ax
        var acy = points[3 * c + 1] - ay
        var acz = points[3 * c + 2] - az
        var nx = aby * acz - abz * acy
        var ny = abz * acx - abx * acz
        var nz = abx * acy - aby * acx
        var norm = sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 0.0:
            dst[3 * t] = nx / norm
            dst[3 * t + 1] = ny / norm
            dst[3 * t + 2] = nz / norm
        else:
            dst[3 * t] = 0.0
            dst[3 * t + 1] = 0.0
            dst[3 * t + 2] = 0.0


@export("mpa_assign")
def mpa_assign(points_addr: Int, centers_addr: Int, labels_addr: Int, n: Int, k: Int) abi("C") -> Float64:
    var points = FPtr(unsafe_from_address=points_addr)
    var centers = FPtr(unsafe_from_address=centers_addr)
    var labels = IPtr(unsafe_from_address=labels_addr)
    var num_blocks = (n + ASSIGN_BLOCK_SIZE - 1) // ASSIGN_BLOCK_SIZE
    def assign_block(block: Int) capturing:
        var start = block * ASSIGN_BLOCK_SIZE
        var stop = min(start + ASSIGN_BLOCK_SIZE, n)
        for i in range(start, stop):
            var dx = points[3 * i] - centers[0]
            var dy = points[3 * i + 1] - centers[1]
            var dz = points[3 * i + 2] - centers[2]
            var best = dx * dx + dy * dy + dz * dz
            var winner = 0
            for cluster in range(1, k):
                dx = points[3 * i] - centers[3 * cluster]
                dy = points[3 * i + 1] - centers[3 * cluster + 1]
                dz = points[3 * i + 2] - centers[3 * cluster + 2]
                var dist = dx * dx + dy * dy + dz * dz
                if dist < best:
                    best = dist
                    winner = cluster
            labels[i] = winner
    for block in range(num_blocks):
        assign_block(block)
    var inertia = 0.0
    for i in range(n):
        var offset = 3 * labels[i]
        var dx = points[3 * i] - centers[offset]
        var dy = points[3 * i + 1] - centers[offset + 1]
        var dz = points[3 * i + 2] - centers[offset + 2]
        inertia += dx * dx + dy * dy + dz * dz
    return inertia


@export("mpa_weighted_update")
def mpa_weighted_update(
    points_addr: Int, weights_addr: Int, labels_addr: Int, centers_addr: Int,
    sums_addr: Int, masses_addr: Int, n: Int, k: Int
) abi("C") -> Float64:
    var points = FPtr(unsafe_from_address=points_addr)
    var weights = FPtr(unsafe_from_address=weights_addr)
    var labels = IPtr(unsafe_from_address=labels_addr)
    var centers = FPtr(unsafe_from_address=centers_addr)
    var sums = FPtr(unsafe_from_address=sums_addr)
    var masses = FPtr(unsafe_from_address=masses_addr)
    for cluster in range(k):
        masses[cluster] = 0.0
        sums[3 * cluster] = 0.0
        sums[3 * cluster + 1] = 0.0
        sums[3 * cluster + 2] = 0.0
    for i in range(n):
        var cluster = labels[i]
        var weight = weights[i]
        masses[cluster] += weight
        sums[3 * cluster] += weight * points[3 * i]
        sums[3 * cluster + 1] += weight * points[3 * i + 1]
        sums[3 * cluster + 2] += weight * points[3 * i + 2]
    var shift = 0.0
    for cluster in range(k):
        if masses[cluster] > 0.0:
            for j in range(3):
                var updated = sums[3 * cluster + j] / masses[cluster]
                var delta = updated - centers[3 * cluster + j]
                shift += delta * delta
                centers[3 * cluster + j] = updated
    return shift

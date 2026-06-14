import numpy as np
import os
import struct


# ---------------------------------------------------------------------------
# Minimal binary STL loader — no external dependencies
# ---------------------------------------------------------------------------
def load_stl_binary(filepath):
    """
    Load a binary STL file and return (vertices, faces) as numpy arrays.

    Args:
        filepath: Path to .stl file (binary format)

    Returns:
        vertices:  (V, 3) float32 array of unique vertices
        triangles: (T, 3) uint32 array of face indices
    """
    with open(filepath, "rb") as f:
        f.read(80)  # skip header
        n_faces = struct.unpack("<I", f.read(4))[0]

        all_verts = []
        for _ in range(n_faces):
            f.read(12)  # skip normal
            for _ in range(3):
                v = struct.unpack("<3f", f.read(12))
                all_verts.append(v)
            f.read(2)  # skip attribute byte count

    raw_verts = np.array(all_verts, dtype=np.float32)
    # Deduplicate vertices and build indexed triangle list
    unique_verts, inverse = np.unique(raw_verts, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.uint32)

    return unique_verts, triangles


# ---------------------------------------------------------------------------
# Box primitive
# ---------------------------------------------------------------------------
def box_mesh(center, size):
    """
    Create a box triangle mesh.

    Args:
        center: [x, y, z] — center position of the box
        size:   [lx, ly, lz] — full dimensions of the box

    Returns:
        vertices:  (8, 3) float32 array
        triangles: (12, 3) uint32 array (2 per face × 6 faces)
    """
    lx, ly, lz = size
    x, y, z = lx / 2.0, ly / 2.0, lz / 2.0

    vertices = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x,  y, -z],
        [-x,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ x,  y,  z],
        [-x,  y,  z],
    ], dtype=np.float32)

    vertices += np.array(center, dtype=np.float32)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom (-z)
        [4, 5, 6], [4, 6, 7],  # top (+z)
        [0, 1, 5], [0, 5, 4],  # front (-y)
        [1, 2, 6], [1, 6, 5],  # right (+x)
        [2, 3, 7], [2, 7, 6],  # back (+y)
        [3, 0, 4], [3, 4, 7],  # left (-x)
    ], dtype=np.uint32)

    return vertices, triangles


def create_bridge_mesh(track_width, track_block_length, wall_thickness,
                       n_beams, beam_width, gap_width, bridge_height, bridge_length,
                       rotate_90=False, ramp_length=0.0):
    """
    Build a pure multi-beam bridge mesh: beams only, filling the entire block.

    No ground plane, no side walls, no approach platforms — only the beams.
    Optionally adds approach ramps (ascending + descending) at the ends.

    Two modes:
      - rotate_90=False (default): Beams run along X (track direction),
        spaced along Y (across track). Beam length = bridge_length along X.
        Pattern repeats along Y to fill track_width.
      - rotate_90=True: Beams run along Y (across track), spanning full
        track_width. Beams are spaced along X with beam_width + gap_width.
        Pattern repeats along X to fill track_block_length.

    When ramp_length > 0, ascending/descending ramps are placed before/after
    the beam pattern, both rising/dropping to bridge_height.

    Coordinate system (Block-local):
      X → forward  (along track), range [0, track_block_length]
      Y → lateral  (across track), range [0, track_width]
      Z → up

    Args:
        track_width:        [m] total block width along Y
        track_block_length: [m] total block length along X
        wall_thickness:     [m] (unused, kept for API compatibility)
        n_beams:            (unused, count is computed to fill the block)
        beam_width:         [m] width of each beam (along Y in normal mode,
                                  along X in rotate_90 mode)
        gap_width:          [m] width of each gap between beams
        bridge_height:      [m] elevation of beams above ground
        bridge_length:      [m] length of beams (along X in normal mode,
                                  not used in rotate_90 mode)
        rotate_90:          If True, beams run along Y instead of X.
        ramp_length:        [m] length of approach ramps (0=no ramps).

    Returns:
        vertices:  (V, 3) float32
        triangles: (T, 3) uint32
    """
    vertices_all = []
    triangles_all = []
    offset = 0

    # Clamp ramp_length so total fits (at most 1/3 of block per side)
    ramp_length = max(0.0, min(ramp_length, track_block_length / 3.0))
    has_ramps = ramp_length > 0.001

    if rotate_90:
        # Beams run along Y (across track), spanning full track_width.
        # Beams are spaced along X with beam_width + gap_width.
        n_beams_fit = int((track_block_length + gap_width) / (beam_width + gap_width))
        if n_beams_fit < 1:
            n_beams_fit = 1

        total_pattern_length = n_beams_fit * beam_width + (n_beams_fit - 1) * gap_width
        beam_area_start_x = (track_block_length - total_pattern_length) / 2.0
        beam_area_end_x = beam_area_start_x + total_pattern_length
        beam_y_center = track_width / 2.0

        # --- Ascending ramp (front): goes from ground up to beam_area_start_x ---
        if has_ramps:
            asc_ramp_len = min(ramp_length, beam_area_start_x)
            asc_start_x = beam_area_start_x - asc_ramp_len
            v, t = create_ramp_mesh(
                ramp_length=asc_ramp_len,
                ramp_width=track_width,
                ramp_height=bridge_height,
                ramp_x_start=asc_start_x,
                ramp_y_start=0.0,
                descend=False,
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- Beams ---
        for i in range(n_beams_fit):
            x_center = beam_area_start_x + i * (beam_width + gap_width) + beam_width / 2.0
            v, t = box_mesh(
                center=[x_center, beam_y_center, bridge_height / 2],
                size=[beam_width, track_width, bridge_height],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- Descending ramp (back): goes from beam_area_end_x down to ground ---
        if has_ramps:
            desc_ramp_len = min(ramp_length, track_block_length - beam_area_end_x)
            v, t = create_ramp_mesh(
                ramp_length=desc_ramp_len,
                ramp_width=track_width,
                ramp_height=bridge_height,
                ramp_x_start=beam_area_end_x,
                ramp_y_start=0.0,
                descend=True,
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)
    else:
        # Beams run along X (track direction), spaced along Y.
        beam_x_start = (track_block_length - bridge_length) / 2.0
        beam_x_end = beam_x_start + bridge_length
        beam_x_center = track_block_length / 2.0

        n_beams_fit = int((track_width + gap_width) / (beam_width + gap_width))
        if n_beams_fit < 1:
            n_beams_fit = 1

        total_pattern_width = n_beams_fit * beam_width + (n_beams_fit - 1) * gap_width
        pattern_start_y = (track_width - total_pattern_width) / 2.0

        # --- Ascending ramp (front): goes from ground up to beam_x_start ---
        if has_ramps:
            asc_ramp_len = min(ramp_length, beam_x_start)
            asc_start_x = beam_x_start - asc_ramp_len
            v, t = create_ramp_mesh(
                ramp_length=asc_ramp_len,
                ramp_width=track_width,
                ramp_height=bridge_height,
                ramp_x_start=asc_start_x,
                ramp_y_start=0.0,
                descend=False,
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- Beams ---
        for i in range(n_beams_fit):
            y_center = pattern_start_y + i * (beam_width + gap_width) + beam_width / 2.0
            v, t = box_mesh(
                center=[beam_x_center, y_center, bridge_height / 2],
                size=[bridge_length, beam_width, bridge_height],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- Descending ramp (back): goes from beam_x_end down to ground ---
        if has_ramps:
            desc_ramp_len = min(ramp_length, track_block_length - beam_x_end)
            v, t = create_ramp_mesh(
                ramp_length=desc_ramp_len,
                ramp_width=track_width,
                ramp_height=bridge_height,
                ramp_x_start=beam_x_end,
                ramp_y_start=0.0,
                descend=True,
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

    if len(vertices_all) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32))

    vertices = np.concatenate(vertices_all, axis=0)
    triangles = np.concatenate(triangles_all, axis=0)

    return vertices.astype(np.float32), triangles.astype(np.uint32)


def create_bridge_a_mesh(track_width, track_block_length, wall_thickness=0.04):
    """
    Bridge A: beams along Y (across-track, rotate_90=True).
      beam_width=40cm (along X), gap_width=15cm (along X), 20cm elevated.
      Beams span full track_width, fill track_block_length.
    """
    return create_bridge_mesh(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=5,
        beam_width=0.40,
        gap_width=0.15,
        bridge_height=0.20,
        bridge_length=2.2,  # unused in rotate_90 mode
        rotate_90=True,
    )


def create_bridge_b_mesh(track_width, track_block_length, wall_thickness=0.04):
    """
    Bridge B: beams along Y (across-track, rotate_90=True).
      beam_width=20cm (along X), gap_width=10cm (along X), 25cm elevated.
      Beams span full track_width, fill track_block_length.
    """
    return create_bridge_mesh(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=3,
        beam_width=0.20,
        gap_width=0.10,
        bridge_height=0.25,
        bridge_length=1.5,  # unused in rotate_90 mode
        rotate_90=True,
    )


# ===========================================================================
# Alternative Implementation 2: Platform-based bridge mesh
# ===========================================================================
# Inspired by the "trench" approach in My_elmap-rl-controller, but inverted
# for elevated bridges: instead of building individual beam boxes UP from
# the ground, this approach builds a single solid platform at bridge_height
# spanning the full beam+gap pattern. The gaps remain empty (no mesh),
# creating the same physical effect with a different mesh topology.
#
# Key difference from create_bridge_mesh():
#   - Beams are connected as one platform piece (fewer vertices)
#   - Same ground plane + side walls + approach platforms
#   - Gap areas have NO mesh at any height (robot falls through)
# ===========================================================================
def create_bridge_mesh_platform(track_width, track_block_length, wall_thickness,
                                 n_beams, beam_width, gap_width, bridge_height,
                                 bridge_length):
    """
    Build bridge mesh using the "platform" approach — a single connected
    platform with empty gaps, rather than individual beam boxes.

    Pure platform only: no ground plane, no side walls, no approach platforms.
    The platform fills track_width along Y and spans bridge_length along X.

    Args:
        track_width, track_block_length, wall_thickness,
        n_beams, beam_width, gap_width, bridge_height, bridge_length:
        Same as create_bridge_mesh().

    Returns:
        vertices:  (V, 3) float32
        triangles: (T, 3) uint32
    """
    vertices_all = []
    triangles_all = []
    offset = 0

    # Compute how many beams fit in track_width
    n_beams_fit = int((track_width + gap_width) / (beam_width + gap_width))
    total_pattern_width = n_beams_fit * beam_width + (n_beams_fit - 1) * gap_width
    beam_x_center = track_block_length / 2.0

    # Single platform covering the entire pattern area
    v, t = box_mesh(
        center=[beam_x_center, track_width / 2, bridge_height / 2],
        size=[bridge_length, total_pattern_width, bridge_height],
    )
    vertices_all.append(v)
    triangles_all.append(t + offset)

    if len(vertices_all) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32))

    vertices = np.concatenate(vertices_all, axis=0)
    triangles = np.concatenate(triangles_all, axis=0)

    return vertices.astype(np.float32), triangles.astype(np.uint32)


# ===========================================================================
# Alternative Implementation 3: STL-file-based bridge mesh
# ===========================================================================
# Loads a pre-made STL 3D model as the bridge surface and combines it with
# procedural ground plane, side walls, and approach platforms.
#
# Inspired by _create_custom_terrain_meshes() in My_elmap-rl-controller,
# which uses trimesh.load() + gym.add_triangle_mesh() to place STL assets.
# ===========================================================================

# Default STL path relative to the project root
_DEFAULT_STL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "resources", "terrains", "Obstacle_Course_Model",
)

# Mapping from bridge type to STL filename
BRIDGE_STL_MAP = {
    "bridge_a": "sloped_wooden_bridge_single.stl",
    "bridge_b": "sloped_wooden_bridge_single.stl",
}

# The combined STL file contains two bridge models stacked along the Y axis.
# We split them by connected-component Y-centers:
#   bridge_a (upper):  5-beam multi-beam bridge, Y roughly [-1.5, 0.5]
#   bridge_b (lower):  3-beam multi-beam bridge, Y roughly [-7.0, -1.5]
_BRIDGE_Y_SPLIT_THRESHOLD = -1.6  # midpoint between the two groups


def _find_connected_components(triangles, num_vertices):
    """
    Partition triangles into connected components via edge-adjacency BFS.

    Args:
        triangles: (T, 3) uint32 array of vertex indices
        num_vertices: total number of vertices (used to size edge map)

    Returns:
        list of lists of triangle indices, one per connected component.
    """
    # Build edge → triangle adjacency
    edge_to_tris = {}
    for tri_idx, t in enumerate(triangles):
        v0, v1, v2 = int(t[0]), int(t[1]), int(t[2])
        for edge in [(v0, v1), (v1, v2), (v2, v0)]:
            edge_sorted = tuple(sorted(edge))
            edge_to_tris.setdefault(edge_sorted, []).append(tri_idx)

    # Triangle adjacency graph
    adj = {i: set() for i in range(len(triangles))}
    for edge, tri_list in edge_to_tris.items():
        for i in range(len(tri_list)):
            for j in range(i + 1, len(tri_list)):
                adj[tri_list[i]].add(tri_list[j])
                adj[tri_list[j]].add(tri_list[i])

    # BFS to find connected components
    visited = set()
    components = []
    for start_tri in range(len(triangles)):
        if start_tri in visited:
            continue
        comp = []
        stack = [start_tri]
        visited.add(start_tri)
        while stack:
            t = stack.pop()
            comp.append(t)
            for neighbor in adj[t]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)

    return components


def _extract_bridge_submesh(vertices, triangles, bridge_type):
    """
    Extract the sub-mesh for bridge_a or bridge_b from the combined STL.

    The combined STL (sloped_wooden_bridge_single.stl) contains both the
    5-beam bridge (bridge_a, in the upper Y region) and the 3-beam bridge
    (bridge_b, in the lower Y region) as disconnected components.

    This function identifies which connected components belong to the
    requested bridge type by comparing their Y-centers against a threshold,
    and returns only those triangles/vertices.

    Args:
        vertices:   (V, 3) float32 — all STL vertices
        triangles:  (T, 3) uint32 — all STL triangle indices
        bridge_type: "bridge_a" or "bridge_b"

    Returns:
        sub_verts:  (V', 3) float32 — vertices for the sub-mesh
        sub_tris:   (T', 3) uint32 — triangle indices (re-indexed)
    """
    if bridge_type not in ("bridge_a", "bridge_b"):
        return vertices, triangles  # no split

    components = _find_connected_components(triangles, len(vertices))

    # Compute Y-center for each component (mean Y of its triangle centers)
    comp_y_centers = []
    for comp in components:
        comp_tris = triangles[comp]
        tri_centers_y = np.mean(vertices[comp_tris][:, :, 1], axis=1)
        comp_y_centers.append(float(np.mean(tri_centers_y)))

    # Assign components to bridge_a (upper) or bridge_b (lower)
    selected_tri_indices = []
    for comp, y_center in zip(components, comp_y_centers):
        if bridge_type == "bridge_a" and y_center >= _BRIDGE_Y_SPLIT_THRESHOLD:
            selected_tri_indices.extend(comp)
        elif bridge_type == "bridge_b" and y_center < _BRIDGE_Y_SPLIT_THRESHOLD:
            selected_tri_indices.extend(comp)

    if len(selected_tri_indices) == 0:
        # Fallback: return empty mesh (shouldn't happen with valid STL)
        print(f"[bridge_mesh] WARNING: No triangles found for {bridge_type}. "
              f"Returning empty sub-mesh.")
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32))

    # Extract the selected triangles and re-index vertices
    selected_tris = triangles[np.array(selected_tri_indices)]
    used_vertex_indices = np.unique(selected_tris.flatten())
    used_verts = vertices[used_vertex_indices]

    # Build old→new index mapping and re-index triangles
    old_to_new = {old: new for new, old in enumerate(used_vertex_indices)}
    reindexed_tris = np.array(
        [[old_to_new[int(v)] for v in t] for t in selected_tris],
        dtype=np.uint32,
    )

    return used_verts.astype(np.float32), reindexed_tris


def _resolve_stl_path(stl_path=None, bridge_type=None):
    """Resolve STL file path from explicit path, bridge_type map, or default dir."""
    if stl_path is not None and os.path.isfile(stl_path):
        return stl_path
    if bridge_type is not None and bridge_type in BRIDGE_STL_MAP:
        default_path = os.path.join(_DEFAULT_STL_DIR, BRIDGE_STL_MAP[bridge_type])
        if os.path.isfile(default_path):
            return default_path
    # Fall back to default dir search
    if stl_path is not None:
        candidate = os.path.join(_DEFAULT_STL_DIR, stl_path)
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Ramp mesh primitive — a wedge from z=0 to z=ramp_height
# ---------------------------------------------------------------------------
def create_ramp_mesh(ramp_length, ramp_width, ramp_height, ramp_x_start=0.0,
                     ramp_y_start=0.0, descend=False):
    """
    Create a sloped ramp (wedge) triangle mesh.

    The ramp goes along +X, spanning ramp_width along Y.
      - descend=False: rises from z=0 at x_start to z=ramp_height at x_start+length
      - descend=True:  drops from z=ramp_height at x_start to z=0 at x_start+length

    Args:
        ramp_length:   [m] ramp extent along X
        ramp_width:    [m] ramp extent along Y
        ramp_height:   [m] height difference from start to end
        ramp_x_start:  [m] X offset for the ramp start
        ramp_y_start:  [m] Y offset for the ramp start
        descend:       If True, ramp goes down (z=H→0); if False, up (z=0→H)

    Returns:
        vertices:  (8, 3) float32
        triangles: (8, 3) uint32
    """
    L = ramp_length
    W = ramp_width
    H = ramp_height
    x0 = ramp_x_start
    x1 = ramp_x_start + L
    y0 = ramp_y_start
    y1 = ramp_y_start + W

    if descend:
        # Ramp drops from z=H at x0 to z=0 at x1
        z_back = H   # high end (back)
        z_front = 0.0  # low end (front)
    else:
        # Ramp rises from z=0 at x0 to z=H at x1
        z_back = 0.0
        z_front = H

    vertices = np.array([
        [x0, y1, z_back],   # 0: back-right
        [x0, y0, z_back],   # 1: back-left
        [x1, y0, z_front],  # 2: front-left
        [x1, y1, z_front],  # 3: front-right
        [x0, y1, z_back],   # 4: degenerate copy of 0
        [x0, y0, z_back],   # 5: degenerate copy of 1
        [x1, y0, z_front],  # 6: degenerate copy of 2
        [x1, y1, z_front],  # 7: degenerate copy of 3
    ], dtype=np.float32)

    # NOTE: The thin edge has degenerate vertices (duplicates).
    # The resulting 8-triangle mesh is watertight for physics.
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top (sloped)
        [1, 2, 6],               # side — single triangle (v1=v5)
        [2, 3, 7], [2, 7, 6],  # front
        [3, 0, 7],               # side — single triangle (v0=v4)
    ], dtype=np.uint32)

    return vertices, triangles


# ---------------------------------------------------------------------------
# Barrier mesh — thin wall to fill gap between bridge pattern and track edges
# ---------------------------------------------------------------------------
def create_barrier_mesh(barrier_length, barrier_height, barrier_thickness,
                        barrier_x_start, barrier_y_center):
    """
    Create a thin barrier wall to prevent the robot from walking around
    the bridge obstacle.

    Args:
        barrier_length:    [m] extent along X
        barrier_height:    [m] height of the barrier
        barrier_thickness: [m] thickness along Y
        barrier_x_start:   [m] X start position
        barrier_y_center:  [m] Y center position

    Returns:
        vertices:  (8, 3) float32
        triangles: (12, 3) uint32
    """
    return box_mesh(
        center=[barrier_x_start + barrier_length / 2.0,
                barrier_y_center,
                barrier_height / 2.0],
        size=[barrier_length, barrier_thickness, barrier_height],
    )


# ---------------------------------------------------------------------------
# Mesh transformation helpers
# ---------------------------------------------------------------------------
def _rotate_mesh_90(vertices):
    """
    Rotate mesh 90 degrees around Z by swapping X and Y coordinates.

    Args:
        vertices: (V, 3) float32

    Returns:
        rotated:  (V, 3) float32 with X and Y swapped
    """
    rotated = vertices.copy()
    rotated[:, 0], rotated[:, 1] = vertices[:, 1].copy(), vertices[:, 0].copy()
    return rotated


def _trim_mesh_x(vertices, triangles, x_min, x_max):
    """
    Keep only triangles whose ALL vertices fall within [x_min, x_max].

    Args:
        vertices:  (V, 3) float32
        triangles: (T, 3) uint32
        x_min:     minimum X to keep
        x_max:     maximum X to keep

    Returns:
        trimmed_verts: (V', 3) float32
        trimmed_tris:  (T', 3) uint32 (re-indexed)
    """
    tri_verts = vertices[triangles]       # (T, 3, 3)
    x_coords = tri_verts[:, :, 0]         # (T, 3)
    keep = (x_coords >= x_min).all(axis=1) & (x_coords <= x_max).all(axis=1)

    if not keep.any():
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32))

    kept_tris = triangles[keep]
    used_idx = np.unique(kept_tris.flatten())
    kept_verts = vertices[used_idx]
    old_to_new = {int(old): new for new, old in enumerate(used_idx)}
    reindexed = np.array(
        [[old_to_new[int(v)] for v in t] for t in kept_tris],
        dtype=np.uint32,
    )

    return kept_verts.astype(np.float32), reindexed


def _find_deck_start_x(vertices, z_threshold_ratio=0.6):
    """
    Find the X coordinate where the bridge deck (elevated surface) begins.

    Scans X from min to max; returns the first X where any vertex has
    Z >= z_threshold_ratio * max_Z.

    Args:
        vertices:          (V, 3) float32
        z_threshold_ratio: fraction of max Z to consider as "deck"

    Returns:
        deck_start_x: float — the X position where the deck begins
    """
    z_max = vertices[:, 2].max()
    threshold = z_max * z_threshold_ratio
    high_verts = vertices[vertices[:, 2] >= threshold]
    if len(high_verts) == 0:
        return vertices[:, 0].min()
    return float(high_verts[:, 0].min())


def create_bridge_mesh_from_stl(track_width, track_block_length, wall_thickness,
                                 n_beams, beam_width, gap_width, bridge_height,
                                 bridge_length, stl_path=None, bridge_type=None):
    """
    Build bridge mesh using an STL file for the bridge deck geometry.

    Pure bridge deck only: no ground plane, no side walls, no ramps,
    no barriers. The STL bridge deck is tiled along Y to fill track_width.

    Args:
        track_width, track_block_length, wall_thickness,
        n_beams, beam_width, gap_width, bridge_height, bridge_length:
            Same as create_bridge_mesh().
        stl_path:       Path to STL file. If None, uses default.
        bridge_type:    "bridge_a" or "bridge_b".

    Returns:
        vertices:  (V, 3) float32
        triangles: (T, 3) uint32
    """
    vertices_all = []
    triangles_all = []
    offset = 0

    # ------------------------------------------------------------------
    # Load STL and extract bridge deck
    # ------------------------------------------------------------------
    resolved_stl = _resolve_stl_path(stl_path, bridge_type)

    if resolved_stl is None:
        # Fallback: use procedural beams
        print(f"[bridge_mesh] WARNING: STL file not found "
              f"(stl_path={stl_path}, bridge_type={bridge_type}). "
              f"Falling back to procedural beams.")
        return create_bridge_mesh(
            track_width, track_block_length, wall_thickness,
            n_beams, beam_width, gap_width, bridge_height, bridge_length,
        )

    stl_verts, stl_tris = load_stl_binary(resolved_stl)
    sub_verts, sub_tris = _extract_bridge_submesh(stl_verts, stl_tris, bridge_type)

    if sub_verts.shape[0] == 0:
        print(f"[bridge_mesh] WARNING: Sub-mesh extraction failed for "
              f"{bridge_type}. Falling back to procedural beams.")
        return create_bridge_mesh(
            track_width, track_block_length, wall_thickness,
            n_beams, beam_width, gap_width, bridge_height, bridge_length,
        )

    # ------------------------------------------------------------------
    # Extract only the elevated deck portion (Z > 30% of max Z)
    # ------------------------------------------------------------------
    z_max = sub_verts[:, 2].max()
    z_threshold = z_max * 0.3
    tri_z = sub_verts[sub_tris][:, :, 2]  # (T, 3)
    elevated_mask = (tri_z >= z_threshold).any(axis=1)
    deck_tris = sub_tris[elevated_mask]

    if len(deck_tris) == 0:
        # No elevated triangles — use all
        deck_tris = sub_tris

    # Re-index
    used_idx = np.unique(deck_tris.flatten())
    deck_verts = sub_verts[used_idx].copy()
    old_to_new = {int(old): new for new, old in enumerate(used_idx)}
    deck_tris = np.array(
        [[old_to_new[int(v)] for v in t] for t in deck_tris],
        dtype=np.uint32,
    )

    # Native extents of the deck
    deck_x_min = deck_verts[:, 0].min()
    deck_x_max = deck_verts[:, 0].max()
    deck_y_min = deck_verts[:, 1].min()
    deck_y_max = deck_verts[:, 1].max()
    deck_z_min = deck_verts[:, 2].min()
    deck_z_max = deck_verts[:, 2].max()

    deck_x_span = deck_x_max - deck_x_min
    deck_y_span = deck_y_max - deck_y_min
    deck_z_span = deck_z_max - deck_z_min

    # ------------------------------------------------------------------
    # Scale: Z to bridge_height, X to fit bridge_length (centered)
    #         Y: keep native proportions, tile to fill track_width
    # ------------------------------------------------------------------
    if deck_x_span < 1e-6 or deck_y_span < 1e-6 or deck_z_span < 1e-6:
        return create_bridge_mesh(
            track_width, track_block_length, wall_thickness,
            n_beams, beam_width, gap_width, bridge_height, bridge_length,
        )

    scale_z = bridge_height / deck_z_span
    # X: scale to fit bridge_length (so it centers nicely), but don't exceed block
    target_x = min(bridge_length, track_block_length)
    scale_x = target_x / deck_x_span
    scale_y = 1.0  # preserve native Y

    # Apply scales relative to min corner
    deck_verts[:, 0] = (deck_verts[:, 0] - deck_x_min) * scale_x
    deck_verts[:, 1] = (deck_verts[:, 1] - deck_y_min) * scale_y
    deck_verts[:, 2] = (deck_verts[:, 2] - deck_z_min) * scale_z

    scaled_y_span = deck_y_span * scale_y
    scaled_x_span = deck_x_span * scale_x

    # ------------------------------------------------------------------
    # Tile the deck along Y to fill track_width
    # ------------------------------------------------------------------
    if scaled_y_span <= 0:
        return create_bridge_mesh(
            track_width, track_block_length, wall_thickness,
            n_beams, beam_width, gap_width, bridge_height, bridge_length,
        )

    n_copies = max(1, int(np.ceil(track_width / scaled_y_span)))
    # Adjust to avoid excessive overlap: compute exact spacing
    y_spacing = track_width / n_copies if n_copies > 1 else scaled_y_span

    for copy_i in range(n_copies):
        copy_verts = deck_verts.copy()
        # Center this copy in its Y slot
        y_slot_center = (copy_i + 0.5) * y_spacing
        y_pattern_center = scaled_y_span / 2.0
        y_offset = y_slot_center - y_pattern_center
        copy_verts[:, 1] += y_offset

        # Center along X
        x_offset = (track_block_length - scaled_x_span) / 2.0
        copy_verts[:, 0] += x_offset

        vertices_all.append(copy_verts)
        triangles_all.append(deck_tris + offset)
        offset += len(copy_verts)

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------
    vertices = np.concatenate(vertices_all, axis=0)
    triangles = np.concatenate(triangles_all, axis=0)

    return vertices.astype(np.float32), triangles.astype(np.uint32)


# ---------------------------------------------------------------------------
# Convenience wrappers for bridge_a / bridge_b using STL source
# ---------------------------------------------------------------------------
def create_bridge_a_mesh_stl(track_width, track_block_length, wall_thickness=0.04):
    """Bridge A using STL mesh (5-beam bridge deck, tiled to fill block)."""
    return create_bridge_mesh_from_stl(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=5,
        beam_width=0.15,
        gap_width=0.40,
        bridge_height=0.20,
        bridge_length=2.2,
        bridge_type="bridge_a",
    )


def create_bridge_b_mesh_stl(track_width, track_block_length, wall_thickness=0.04):
    """Bridge B using STL mesh (3-beam bridge deck, tiled to fill block)."""
    return create_bridge_mesh_from_stl(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=3,
        beam_width=0.10,
        gap_width=0.20,
        bridge_height=0.25,
        bridge_length=1.5,
        bridge_type="bridge_b",
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for bridge_a / bridge_b using platform source
# ---------------------------------------------------------------------------
def create_bridge_a_mesh_platform(track_width, track_block_length, wall_thickness=0.04):
    """Bridge A using platform mesh (single platform + empty gaps)."""
    return create_bridge_mesh_platform(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=5,
        beam_width=0.15,
        gap_width=0.40,
        bridge_height=0.20,
        bridge_length=2.2,
    )


def create_bridge_b_mesh_platform(track_width, track_block_length, wall_thickness=0.04):
    """Bridge B using platform mesh (single platform + empty gaps)."""
    return create_bridge_mesh_platform(
        track_width=track_width,
        track_block_length=track_block_length,
        wall_thickness=wall_thickness,
        n_beams=3,
        beam_width=0.10,
        gap_width=0.20,
        bridge_height=0.25,
        bridge_length=1.5,
    )


# ===========================================================================
# I-stairs mesh support
# ===========================================================================

def create_istairs_mesh(track_width, track_block_length, wall_thickness,
                         step_height, step_depth, n_steps,
                         stair_width, platform_width=0.8,
                         rotate_90=False):
    """
    Build a complete I-stairs mesh: ascending stairs → platform → descending stairs.

    No ground plane, no side walls — only the step boxes + center platform.
    The stairs go up from ground, hit a platform, then go back down to ground.

    Args:
        track_width:        [m] total block width along Y
        track_block_length: [m] total block length along X
        wall_thickness:     [m] (unused, kept for API compatibility)
        step_height:        [m] height of each step
        step_depth:         [m] depth of each step along the step direction
        n_steps:            number of steps on EACH side (ascending + descending)
        stair_width:        [m] width of stairs (perpendicular to step direction)
        platform_width:     [m] depth of the top platform
        rotate_90:          If True, steps run along Y instead of X.

    Returns:
        vertices:  (V, 3) float32
        triangles: (T, 3) uint32

    Layout (when rotate_90=False, steps along X):
        ┌──────────────────────────────────────────────┐
        │  ascending stairs  │  platform  │ descending  │
        │     (n steps)      │            │  (n steps)  │
        └──────────────────────────────────────────────┘
    """
    vertices_all = []
    triangles_all = []
    offset = 0

    # Clamp stair width to usable space
    usable_width = track_width - 2 * wall_thickness
    actual_width = min(stair_width, usable_width)

    n = int(n_steps)
    total_height = n * step_height
    stairs_depth = step_depth * n
    # Total: up-stairs + platform + down-stairs
    total_depth = stairs_depth * 2 + platform_width

    # Clamp n_steps if the total doesn't fit in the block
    max_steps = int((track_block_length - platform_width) / (2 * step_depth)) if step_depth > 0 else n
    if max_steps < 1:
        max_steps = 1
    if n > max_steps:
        n = max_steps
        stairs_depth = step_depth * n
        total_depth = stairs_depth * 2 + platform_width
        total_height = n * step_height

    if rotate_90:
        # Steps along Y, spanning X
        step_span_x = actual_width
        step_span_y = step_depth

        y_start = (track_block_length - total_depth) / 2.0
        x_center = track_width / 2.0

        # --- ascending stairs (first n steps, going up) ---
        # Each step is a solid block from z=0 up to its top surface,
        # so there is no hollow space under the stairs.
        for i in range(n):
            y_center = y_start + i * step_depth + step_depth / 2.0
            step_top = (i + 1) * step_height
            v, t = box_mesh(
                center=[x_center, y_center, step_top / 2.0],
                size=[step_span_x, step_span_y, step_top],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- platform at the top ---
        # Solid block from z=0 to total_height (already full-height).
        if platform_width > 0.01:
            plat_y_center = y_start + stairs_depth + platform_width / 2.0
            v, t = box_mesh(
                center=[x_center, plat_y_center, total_height / 2.0],
                size=[step_span_x, platform_width, total_height],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- descending stairs (last n steps, going down) ---
        # Each step is a solid block from z=0 up to its top surface.
        for i in range(n):
            y_center = y_start + stairs_depth + platform_width + i * step_depth + step_depth / 2.0
            step_top = (n - i) * step_height
            v, t = box_mesh(
                center=[x_center, y_center, step_top / 2.0],
                size=[step_span_x, step_span_y, step_top],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

    else:
        # Steps along X, spanning Y
        step_span_x = step_depth
        step_span_y = actual_width

        x_start = (track_block_length - total_depth) / 2.0
        y_center = track_width / 2.0

        # --- ascending stairs (first n steps, going up) ---
        # Each step is a solid block from z=0 up to its top surface,
        # so there is no hollow space under the stairs.
        for i in range(n):
            x_center = x_start + i * step_depth + step_depth / 2.0
            step_top = (i + 1) * step_height
            v, t = box_mesh(
                center=[x_center, y_center, step_top / 2.0],
                size=[step_span_x, step_span_y, step_top],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- platform at the top ---
        # Solid block from z=0 to total_height (already full-height).
        if platform_width > 0.01:
            plat_x_center = x_start + stairs_depth + platform_width / 2.0
            v, t = box_mesh(
                center=[plat_x_center, y_center, total_height / 2.0],
                size=[platform_width, step_span_y, total_height],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

        # --- descending stairs (last n steps, going down) ---
        # Each step is a solid block from z=0 up to its top surface.
        for i in range(n):
            x_center = x_start + stairs_depth + platform_width + i * step_depth + step_depth / 2.0
            step_top = (n - i) * step_height
            v, t = box_mesh(
                center=[x_center, y_center, step_top / 2.0],
                size=[step_span_x, step_span_y, step_top],
            )
            vertices_all.append(v)
            triangles_all.append(t + offset)
            offset += len(v)

    if len(vertices_all) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32))

    vertices = np.concatenate(vertices_all, axis=0)
    triangles = np.concatenate(triangles_all, axis=0)

    return vertices.astype(np.float32), triangles.astype(np.uint32)

"""
Plain-data scene format shared between Person 4 (producer) and the renderer.

Only flat numeric arrays are used, so the same scene can be handed to a C++
backend (pybind11 / numpy buffers) without any change.

A Mesh can contain filled triangles, line segments (edges) or both. Person 4's
current data (vertices + edges) maps directly onto the edge part.

Winding convention
------------------
A triangle is FRONT-facing when its vertices appear counter-clockwise ON THE
SCREEN. The team's camera space is left-handed (x right, y up, z forward, see
math_module.look_at_rotation), so a mesh seen from outside must be listed
clockwise by the right-hand rule. The BOX/PYRAMID constants below already do
this. Culling is off by default, so wrong winding only matters if you turn
`cull_backfaces` on.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np


@dataclass
class Mesh:
    """
    vertices  : (N, 3) float  world-space positions
    triangles : (M, 3) int    indices into vertices (may be empty)
    colors    : (M, 3) uint8  one RGB colour per triangle, or a single (3,)
    edges     : (K, 2) int    line segments as vertex index pairs (optional)
    edge_color: (K, 3) uint8  per-edge colour, or a single (3,) (default white)
    """
    vertices: np.ndarray
    triangles: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    edge_color: Optional[np.ndarray] = None

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (N, 3)")
        n = len(self.vertices)

        tri = (np.zeros((0, 3)) if self.triangles is None
               else np.asarray(self.triangles))
        self.triangles = np.ascontiguousarray(tri.reshape(-1, 3), dtype=np.int32)
        self.colors = self._colours(self.colors, len(self.triangles),
                                    (200, 200, 200), "colors")

        ed = (np.zeros((0, 2)) if self.edges is None
              else np.asarray(self.edges))
        self.edges = np.ascontiguousarray(ed.reshape(-1, 2), dtype=np.int32)
        self.edge_color = self._colours(self.edge_color, len(self.edges),
                                        (255, 255, 255), "edge_color")

        for name, idx in (("triangle", self.triangles), ("edge", self.edges)):
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"{name} index out of range")

    @staticmethod
    def _colours(c, count, default, name):
        c = np.asarray(default if c is None else c, dtype=np.uint8)
        if c.ndim == 1:                       # one colour for everything
            c = np.tile(c, (count, 1))
        if c.shape != (count, 3):
            raise ValueError(f"{name} must be (3,) or ({count}, 3)")
        return np.ascontiguousarray(c)


@dataclass
class Scene:
    meshes: List[Mesh] = field(default_factory=list)

    def add(self, mesh: Mesh) -> "Scene":
        self.meshes.append(mesh)
        return self

    def triangle_count(self) -> int:
        return sum(len(m.triangles) for m in self.meshes)

    def edge_count(self) -> int:
        return sum(len(m.edges) for m in self.meshes)


# ---------------------------------------------------------------------------
# Topology of the team's primitives (vertex order as in scene_cube.py and
# scene_pyramid.py), wound for the team's left-handed screen convention.
# ---------------------------------------------------------------------------

def _flip(t):
    """Reverse the winding of every triangle."""
    return np.ascontiguousarray(np.asarray(t, np.int32)[:, ::-1])


_BOX_TRIS_RH = [
    (4, 5, 6), (4, 6, 7),   # +Z
    (0, 3, 2), (0, 2, 1),   # -Z
    (1, 2, 6), (1, 6, 5),   # +X
    (0, 4, 7), (0, 7, 3),   # -X
    (3, 7, 6), (3, 6, 2),   # +Y
    (0, 1, 5), (0, 5, 4),   # -Y
]
BOX_TRIANGLES = _flip(_BOX_TRIS_RH)

_PYRAMID_TRIS_RH = [(1, 0, 4), (2, 1, 4), (3, 2, 4), (0, 3, 4),   # sides
                    (0, 1, 2), (0, 2, 3)]                         # bottom
PYRAMID_TRIANGLES = _flip(_PYRAMID_TRIS_RH)

BOX_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                      (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)], np.int32)

_BOX_UNIT = np.array([
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
], dtype=np.float32) * 0.5


# ---------------------------------------------------------------------------
# Helpers Person 4 can use to turn "objects, positions, sizes, colors"
# into meshes without knowing anything about rendering.
# ---------------------------------------------------------------------------

def make_box(center: Sequence[float], size: Sequence[float],
             color: Sequence[int], edge_color=None) -> Mesh:
    """Axis-aligned box. `size` = (sx, sy, sz); `color` = RGB 0..255."""
    verts = _BOX_UNIT * np.asarray(size, np.float32) + np.asarray(center, np.float32)
    return Mesh(verts, BOX_TRIANGLES.copy(), np.asarray(color, np.uint8),
                BOX_EDGES.copy() if edge_color is not None else None, edge_color)


def make_ground(half_extent: float, y: float, color: Sequence[int],
                color2: Sequence[int] = None, tiles: int = 1) -> Mesh:
    """Flat square ground at height y, optionally a checkerboard of tiles x tiles."""
    n = tiles
    xs = np.linspace(-half_extent, half_extent, n + 1, dtype=np.float32)
    verts, tris, cols = [], [], []
    for i in range(n):
        for j in range(n):
            b = len(verts)
            x0, x1, z0, z1 = xs[i], xs[i + 1], xs[j], xs[j + 1]
            verts += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
            tris += [(b, b + 3, b + 2), (b, b + 2, b + 1)]
            c = color if (color2 is None or (i + j) % 2 == 0) else color2
            cols += [c, c]
    return Mesh(np.array(verts), _flip(tris), np.array(cols))


def make_pyramid(center_base: Sequence[float], base: float, height: float,
                 color: Sequence[int]) -> Mesh:
    """Square pyramid (useful for roofs / trees)."""
    cx, cy, cz = center_base
    h = base / 2
    v = np.array([(cx - h, cy, cz - h), (cx + h, cy, cz - h),
                  (cx + h, cy, cz + h), (cx - h, cy, cz + h),
                  (cx, cy + height, cz)], np.float32)
    return Mesh(v, PYRAMID_TRIANGLES.copy(), np.asarray(color, np.uint8))

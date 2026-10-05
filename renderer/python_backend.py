"""
Pure-Python (numpy) software rasteriser implementing `Renderer`.

Pipeline (one pass of render()):
  1. gather triangles and edges of all meshes     (world space)
  2. flat shading colour per triangle             (face normal . light)
  3. clip-space transform  clip = P @ V @ v
  4. trivial reject against the view frustum
  5. near-plane clipping (triangles: Sutherland-Hodgman, lines: parametric)
  6. perspective divide -> NDC -> viewport transform (screen pixels)
  7. rasterisation: barycentric triangles + sampled lines, both z-buffered
  8. framebuffer / depth buffer ready for get_framebuffer()

Only clip space is assumed (x/w, y/w, z/w in [-1, 1], w > 0 in front of the
camera), so any handedness / camera axis convention works as long as the
matrices agree with each other.
"""

import math
import numpy as np

from .api import Renderer
from .scene import Scene, Mesh

_EPS = 1e-9
_LINE_DEPTH_BIAS = 2e-4     # lets edges win against the faces they outline


def _as_matrix(m, name):
    a = np.asarray(m, dtype=np.float64)
    if a.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


class PythonRenderer(Renderer):
    BACKEND_NAME = "python"

    def __init__(self, width: int, height: int):
        self._opts = dict(cull_backfaces=False, lighting=True,
                          light_dir=(-0.4, -1.0, -0.6), ambient=0.35,
                          wireframe=False, line_width=1)
        self._clear = np.array([135, 190, 235], np.uint8)   # sky blue
        self._view = np.eye(4)
        self._proj = np.eye(4)
        self._scene = Scene()
        self._stats = {}
        self._allocate(width, height)

    # ------------------------------------------------------------------ setup
    def _allocate(self, width, height):
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self._w, self._h = width, height
        self._fb = np.empty((height, width, 3), np.uint8)
        self._fb[:] = self._clear
        self._zb = np.full((height, width), np.inf, np.float32)

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    def resize(self, width, height):
        self._allocate(width, height)

    def set_view_matrix(self, matrix):
        self._view = _as_matrix(matrix, "view matrix")

    def set_projection_matrix(self, matrix):
        self._proj = _as_matrix(matrix, "projection matrix")

    def set_scene(self, scene):
        if isinstance(scene, Mesh):
            scene = Scene([scene])
        elif not isinstance(scene, Scene):
            scene = Scene(list(scene))        # accept a plain list of meshes
        self._scene = scene

    def set_clear_color(self, rgb):
        self._clear = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)

    def set_options(self, **options):
        for k, v in options.items():
            if k in self._opts:
                self._opts[k] = v             # unknown keys are ignored

    # ----------------------------------------------------------------- render
    def render(self):
        self._fb[:] = self._clear
        self._zb[:] = np.inf
        st = dict(triangles_in=0, triangles_offscreen=0, triangles_culled=0,
                  triangles_clipped=0, triangles_drawn=0, pixels_written=0,
                  lines_in=0, lines_drawn=0)
        self._stats = st

        m = self._proj @ self._view
        tri_w, tri_c, line_w, line_c = self._gather()
        st["triangles_in"] = len(tri_w)
        st["lines_in"] = len(line_w)

        if len(tri_w):
            shaded = self._shade(tri_w, tri_c)
            if self._opts["wireframe"]:
                # outline mode: triangles become edges, no filled faces
                a = np.concatenate([tri_w[:, 0], tri_w[:, 1], tri_w[:, 2]])
                b = np.concatenate([tri_w[:, 1], tri_w[:, 2], tri_w[:, 0]])
                line_w = np.concatenate(
                    [line_w, np.stack([a, b], axis=1)]) if len(line_w) else \
                    np.stack([a, b], axis=1)
                line_c = np.concatenate([line_c, np.tile(shaded, (3, 1))]
                                        ).astype(np.uint8) if len(line_c) else \
                    np.tile(shaded, (3, 1)).astype(np.uint8)
                st["lines_in"] = len(line_w)
            else:
                self._draw_triangles(tri_w, shaded, m, st)
        if len(line_w):
            self._draw_lines(line_w, line_c, m, st)

    # -------------------------------------------------------------- internals
    def _gather(self):
        tw, tc, lw, lc = [], [], [], []
        for mesh in self._scene.meshes:
            v = mesh.vertices.astype(np.float64)
            if len(mesh.triangles):
                tw.append(v[mesh.triangles])
                tc.append(mesh.colors)
            if len(mesh.edges):
                lw.append(v[mesh.edges])
                lc.append(mesh.edge_color)
        cat = lambda xs, shape, dt: np.concatenate(xs) if xs else np.zeros(shape, dt)
        return (cat(tw, (0, 3, 3), float), cat(tc, (0, 3), np.uint8),
                cat(lw, (0, 2, 3), float), cat(lc, (0, 3), np.uint8))

    def _shade(self, tri_w, tri_c):
        """Flat shading: ambient + diffuse, two-sided."""
        if not self._opts["lighting"]:
            return tri_c.astype(np.float64)
        e1 = tri_w[:, 1] - tri_w[:, 0]
        e2 = tri_w[:, 2] - tri_w[:, 0]
        nrm = np.cross(e1, e2)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, _EPS)
        ld = np.asarray(self._opts["light_dir"], np.float64)
        ld = -ld / max(np.linalg.norm(ld), _EPS)           # towards the light
        diff = np.abs(nrm @ ld)[:, None]
        amb = float(self._opts["ambient"])
        return tri_c.astype(np.float64) * (amb + (1.0 - amb) * diff)

    def _to_clip(self, pts_w, m):
        hom = np.concatenate([pts_w, np.ones(pts_w.shape[:-1] + (1,))], axis=-1)
        return hom @ m.T

    # ---- triangles ---------------------------------------------------------
    def _draw_triangles(self, tri_w, shaded, m, st):
        clip = self._to_clip(tri_w, m)                              # (n,3,4)
        x, y, z, w = clip[..., 0], clip[..., 1], clip[..., 2], clip[..., 3]
        outside = (
            np.all(x < -w, axis=1) | np.all(x > w, axis=1) |
            np.all(y < -w, axis=1) | np.all(y > w, axis=1) |
            np.all(z < -w, axis=1) | np.all(z > w, axis=1)
        )
        st["triangles_offscreen"] = int(outside.sum())

        for i in np.nonzero(~outside)[0]:
            poly = self._clip_near(clip[i])
            if len(poly) < 3:
                st["triangles_offscreen"] += 1
                continue
            if len(poly) != 3 or np.any(clip[i, :, 2] + clip[i, :, 3] < 0):
                st["triangles_clipped"] += 1

            ndc = poly[:, :3] / poly[:, 3:4]
            sx = (ndc[:, 0] * 0.5 + 0.5) * self._w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self._h
            sz = ndc[:, 2]
            rgb = np.clip(shaded[i], 0, 255).astype(np.uint8)

            for k in range(1, len(poly) - 1):            # fan-triangulate
                idx = [0, k, k + 1]
                res = self._raster_triangle(sx[idx], sy[idx], sz[idx], rgb)
                if res is None:
                    st["triangles_culled"] += 1
                else:
                    st["triangles_drawn"] += 1
                    st["pixels_written"] += res

    @staticmethod
    def _clip_near(tri):
        """Clip one triangle (3,4) against the near plane z >= -w."""
        d = tri[:, 2] + tri[:, 3]                  # signed distance to plane
        out = []
        for a in range(3):
            b = (a + 1) % 3
            pa, pb, da, db = tri[a], tri[b], d[a], d[b]
            if da >= 0:
                out.append(pa)
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                out.append(pa + t * (pb - pa))
        out = [p for p in out if p[3] > _EPS]       # guard divide by w ~ 0
        return np.array(out) if out else np.zeros((0, 4))

    def _raster_triangle(self, xs, ys, zs, rgb):
        """Returns pixels written, or None if the triangle was culled."""
        x0, x1, x2 = xs
        y0, y1, y2 = ys
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            return None                                     # degenerate
        # y is flipped in screen space => CCW (front) triangles have area < 0
        if self._opts["cull_backfaces"] and area > 0:
            return None

        bx0 = max(0, int(math.floor(xs.min())))
        bx1 = min(self._w - 1, int(math.ceil(xs.max())))
        by0 = max(0, int(math.floor(ys.min())))
        by1 = min(self._h - 1, int(math.ceil(ys.max())))
        if bx0 > bx1 or by0 > by1:
            return 0

        px = np.arange(bx0, bx1 + 1) + 0.5
        py = (np.arange(by0, by1 + 1) + 0.5)[:, None]

        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        eps = -1e-7
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        depth = (w0 * zs[0] + w1 * zs[1] + w2 * zs[2]).astype(np.float32)
        inside &= (depth >= -1.0) & (depth <= 1.0)          # near/far planes

        zb = self._zb[by0:by1 + 1, bx0:bx1 + 1]
        visible = inside & (depth < zb)
        count = int(visible.sum())
        if count:
            zb[visible] = depth[visible]
            self._fb[by0:by1 + 1, bx0:bx1 + 1][visible] = rgb
        return count

    # ---- lines -------------------------------------------------------------
    def _draw_lines(self, line_w, line_c, m, st):
        clip = self._to_clip(line_w, m)                             # (E,2,4)
        for i in range(len(clip)):
            seg = self._clip_line_near(clip[i])
            if seg is None:
                continue
            ndc = seg[:, :3] / seg[:, 3:4]
            sx = (ndc[:, 0] * 0.5 + 0.5) * self._w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self._h
            rgb = np.clip(line_c[i], 0, 255).astype(np.uint8)
            n = self._raster_line(sx, sy, ndc[:, 2], rgb)
            if n is not None:
                st["lines_drawn"] += 1
                st["pixels_written"] += n

    @staticmethod
    def _clip_line_near(seg):
        """Clip a segment (2,4) against the near plane; None if fully behind."""
        d = seg[:, 2] + seg[:, 3]
        if d[0] < 0 and d[1] < 0:
            return None
        out = seg.copy()
        if d[0] < 0:
            out[0] = seg[0] + (d[0] / (d[0] - d[1])) * (seg[1] - seg[0])
        elif d[1] < 0:
            out[1] = seg[1] + (d[1] / (d[1] - d[0])) * (seg[0] - seg[1])
        if np.any(out[:, 3] <= _EPS):
            return None
        return out

    def _raster_line(self, xs, ys, zs, rgb):
        """Liang-Barsky clip to the screen, then sample the segment."""
        W, H = self._w, self._h
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, xs[0]), (dx, W - xs[0]),
                     (-dy, ys[0]), (dy, H - ys[0])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None

        ax, ay = xs[0] + t0 * dx, ys[0] + t0 * dy
        bx, by = xs[0] + t1 * dx, ys[0] + t1 * dy
        za = zs[0] + t0 * (zs[1] - zs[0])
        zb_ = zs[0] + t1 * (zs[1] - zs[0])
        n = int(max(abs(bx - ax), abs(by - ay))) + 2
        t = np.linspace(0.0, 1.0, n)
        px = np.floor(ax + t * (bx - ax)).astype(np.int64)
        py = np.floor(ay + t * (by - ay)).astype(np.int64)
        pz = (za + t * (zb_ - za)).astype(np.float32)

        width = max(1, int(self._opts["line_width"]))
        lo = -(width // 2)
        written = 0
        for oy in range(lo, lo + width):
            for ox in range(lo, lo + width):
                qx, qy = px + ox, py + oy
                ok = ((qx >= 0) & (qx < W) & (qy >= 0) & (qy < H) &
                      (pz >= -1.0) & (pz <= 1.0))
                qx, qy, qz = qx[ok], qy[ok], pz[ok]
                ok = qz < self._zb[qy, qx] + _LINE_DEPTH_BIAS
                qx, qy, qz = qx[ok], qy[ok], qz[ok]
                self._zb[qy, qx] = np.minimum(self._zb[qy, qx], qz)
                self._fb[qy, qx] = rgb
                written += len(qx)
        return written

    # ---------------------------------------------------------------- outputs
    def get_framebuffer(self):
        return self._fb.copy()

    def get_depthbuffer(self):
        return self._zb.copy()

    def get_stats(self):
        return dict(self._stats)

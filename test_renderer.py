"""
Contract + integration tests for Person 3's renderer.

Run from the repository root:   python -m pytest -q

The contract tests only use the public `renderer` API, so the very same file
must pass for a future C++ backend:   RENDERER_BACKEND=cpp python -m pytest
"""

import numpy as np
import pytest

import math_module                                   # Person 1
import scene_cube                                    # Person 4
from renderer import (create_renderer, Scene, Mesh, make_box, make_ground,
                      make_pyramid)
from integration import (view_matrix, perspective_matrix,
                         orthographic_matrix, focal_length_pixels,
                         build_team_scene, projection_from_scene_module)


def look_from(eye, target=(0, 0, 0)):
    return view_matrix(np.array(eye, float), np.array(target, float))


@pytest.fixture
def r():
    """64x48 black canvas, camera at z=-5 looking along +z (team default)."""
    rr = create_renderer(64, 48)
    rr.set_projection_matrix(perspective_matrix(60, 64 / 48, 0.1, 50))
    rr.set_view_matrix(look_from((0, 0, -5)))
    rr.set_clear_color((0, 0, 0))
    rr.set_options(lighting=False)
    return rr


# ----------------------------------------------------------------- contract
def test_api_shapes(r):
    r.set_scene(Scene([make_box((0, 0, 0), (1, 1, 1), (255, 0, 0))]))
    r.render()
    assert r.get_framebuffer().shape == (48, 64, 3)
    assert r.get_framebuffer().dtype == np.uint8
    assert r.get_depthbuffer().shape == (48, 64)
    assert (r.width, r.height) == (64, 48)


def test_empty_scene_is_background(r):
    r.set_scene(Scene())
    r.render()
    assert not r.get_framebuffer().any()


def test_box_in_centre_and_background_corner(r):
    r.set_scene(Scene([make_box((0, 0, 0), (2, 2, 2), (255, 0, 0))]))
    r.render()
    fb = r.get_framebuffer()
    assert tuple(fb[24, 32]) == (255, 0, 0)
    assert tuple(fb[0, 0]) == (0, 0, 0)


def test_depth_ordering(r):
    near = make_box((0, 0, -2), (1, 1, 1), (0, 255, 0))     # closer to camera
    far = make_box((0, 0, 2), (3, 3, 1), (0, 0, 255))
    for order in ([near, far], [far, near]):                # draw order is irrelevant
        r.set_scene(Scene(order))
        r.render()
        assert tuple(r.get_framebuffer()[24, 32]) == (0, 255, 0)


def test_object_behind_camera_not_drawn(r):
    r.set_scene(Scene([make_box((0, 0, -20), (2, 2, 2), (255, 0, 0))]))
    r.render()
    assert not r.get_framebuffer().any()


def test_object_beyond_far_plane_not_drawn(r):
    r.set_scene(Scene([make_box((0, 0, 200), (2, 2, 2), (255, 0, 0))]))
    r.render()
    assert not r.get_framebuffer().any()


def test_near_plane_clipping_of_triangles(r):
    # the ground passes under the camera -> triangles straddle the near plane
    r.set_scene(Scene([make_ground(30, -1, (200, 200, 200))]))
    r.render()
    fb = r.get_framebuffer()
    assert fb[-1].any() and not fb[0].any()
    assert r.get_stats()["triangles_clipped"] > 0


def test_near_plane_clipping_of_lines(r):
    # one end of the edge is behind the camera
    m = Mesh(np.array([[0, -1, -20.0], [0, -1, 10.0]]), edges=[[0, 1]],
             edge_color=(255, 255, 255))
    r.set_scene(Scene([m]))
    r.render()
    assert r.get_framebuffer().any()
    assert r.get_stats()["lines_drawn"] == 1


def test_x_axis_maps_to_screen_right(r):
    r.set_scene(Scene([make_box((2, 0, 0), (1, 1, 1), (255, 255, 255))]))
    r.render()
    cols = np.nonzero(r.get_framebuffer().any(axis=(0, 2)))[0]
    assert cols.min() > 32


def test_y_axis_maps_to_screen_up(r):
    r.set_scene(Scene([make_box((0, 2, 0), (1, 1, 1), (255, 255, 255))]))
    r.render()
    rows = np.nonzero(r.get_framebuffer().any(axis=(1, 2)))[0]
    assert rows.max() < 24


def test_backface_culling_hides_inside_of_box(r):
    r.set_scene(Scene([make_box((0, 0, 0), (100, 100, 100), (9, 9, 9))]))
    r.set_options(cull_backfaces=True)       # camera is inside: all faces point away
    r.render()
    assert not r.get_framebuffer().any()


def test_backface_culling_keeps_outside_of_box(r):
    # catches wrong triangle winding: front faces must survive culling
    for mesh in (make_box((0, 0, 0), (2, 2, 2), (255, 0, 0)),
                 make_pyramid((0, -1, 0), 2, 2, (255, 0, 0))):
        r.set_scene(Scene([mesh]))
        r.set_options(cull_backfaces=True)
        r.render()
        assert tuple(r.get_framebuffer()[24, 32]) == (255, 0, 0)
        assert r.get_stats()["triangles_culled"] > 0      # back faces went away


def test_lighting_gives_faces_different_shades():
    rr = create_renderer(64, 64)
    rr.set_projection_matrix(perspective_matrix(60, 1, 0.1, 50))
    rr.set_view_matrix(look_from((3, 4, -5)))
    rr.set_clear_color((0, 0, 0))
    rr.set_scene(Scene([make_box((0, 0, 0), (2, 2, 2), (200, 200, 200))]))
    rr.render()
    shades = np.unique(rr.get_framebuffer().reshape(-1, 3), axis=0)
    assert len(shades) >= 3                  # background + at least 2 face tones


def test_edge_only_mesh_draws_lines(r):
    m = Mesh(np.array([[-1, 0, 0], [1, 0, 0.0]]), edges=[[0, 1]],
             edge_color=(255, 0, 0))
    r.set_scene(Scene([m]))
    r.render()
    fb = r.get_framebuffer()
    assert tuple(fb[24, 32]) == (255, 0, 0)
    assert r.get_stats()["triangles_in"] == 0


def test_wireframe_option_has_no_filled_faces():
    r = create_renderer(200, 150)
    r.set_projection_matrix(perspective_matrix(60, 200 / 150, 0.1, 50))
    r.set_view_matrix(look_from((0, 0, -5)))
    r.set_clear_color((0, 0, 0))
    r.set_options(lighting=False)
    r.set_scene(Scene([make_box((0, 0, 0), (2, 2, 2), (255, 0, 0))]))
    r.render()
    solid = int(r.get_framebuffer().any(axis=2).sum())
    r.set_options(wireframe=True)
    r.render()
    wire = int(r.get_framebuffer().any(axis=2).sum())
    assert 0 < wire < 0.3 * solid            # only outlines are drawn


def test_bad_matrix_rejected(r):
    with pytest.raises(ValueError):
        r.set_view_matrix(np.eye(3))
    with pytest.raises(ValueError):
        r.set_projection_matrix(np.full((4, 4), np.nan))


def test_unknown_option_ignored(r):
    r.set_options(some_future_option=123)


def test_resize(r):
    r.resize(80, 60)
    r.render()
    assert r.get_framebuffer().shape == (60, 80, 3)


def test_mesh_winding_matches_team_convention():
    """Left-handed space: outward normal = -(e1 x e2)."""
    for mesh, centre in [
        (make_box((1, 2, 3), (2, 4, 6), (1, 1, 1)), np.array([1, 2, 3.0])),
        (make_pyramid((0, 0, 0), 2, 3, (1, 1, 1)), np.array([0, 0.8, 0.0])),
    ]:
        tri = mesh.vertices[mesh.triangles].astype(float)
        n = -np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_centre = tri.mean(axis=1)
        assert np.all(np.einsum("ij,ij->i", n, face_centre - centre) > 0)


def test_mesh_validation():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 5]]), (1, 1, 1))
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 3)), edges=[[0, 9]])


# -------------------------------------------------- integration with team code
def test_pixels_match_person1_projection():
    """
    Every cube vertex must land on the pixel that Person 1's own
    math_module.project_vertices() predicts (same camera, same focal length).
    """
    cfg = scene_cube
    W, H = cfg.screen_width, cfg.screen_height
    rr = create_renderer(W, H)
    rr.set_clear_color((0, 0, 0))
    rr.set_view_matrix(view_matrix(cfg.camera_position, cfg.camera_target))
    rr.set_projection_matrix(projection_from_scene_module(cfg))
    rr.set_scene(Scene([Mesh(cfg.vertices, edges=np.array(cfg.edges),
                             edge_color=(255, 0, 0))]))
    rr.render()
    fb = rr.get_framebuffer()

    f_px = focal_length_pixels(cfg.field_of_view, H)
    xy = math_module.project_vertices(cfg.vertices, cfg.camera_position,
                                      cfg.camera_target, f_px)
    for x, y in xy:
        px, py = int(np.floor(W / 2 + x)), int(np.floor(H / 2 - y))
        patch = fb[py - 1:py + 2, px - 1:px + 2]
        assert (patch == (255, 0, 0)).all(axis=2).any(), (x, y)


def test_team_scene_renders_both_objects():
    cfg = scene_cube
    rr = create_renderer(cfg.screen_width, cfg.screen_height)
    rr.set_clear_color((0, 0, 0))
    rr.set_view_matrix(view_matrix(cfg.camera_position, cfg.camera_target))
    rr.set_projection_matrix(projection_from_scene_module(cfg))
    rr.set_scene(build_team_scene())
    rr.render()
    st = rr.get_stats()
    assert st["triangles_in"] == 12 + 6 and st["lines_in"] == 12 + 8
    assert st["triangles_drawn"] == 18 and st["lines_drawn"] == 20
    ys, xs = np.nonzero(rr.get_framebuffer().any(axis=2))
    assert xs.min() > 0 and xs.max() < cfg.screen_width - 1   # fully on screen
    assert ys.min() > 0 and ys.max() < cfg.screen_height - 1


def test_orthographic_has_no_perspective_shrinking():
    """In orthographic mode the same box has the same size at any distance."""
    rr = create_renderer(100, 100)
    rr.set_clear_color((0, 0, 0))
    rr.set_options(lighting=False)
    rr.set_projection_matrix(orthographic_matrix(-5, 5, -5, 5, 0.1, 100))
    widths = []
    for z in (2.0, 20.0):
        rr.set_view_matrix(look_from((0, 0, -5)))
        rr.set_scene(Scene([make_box((0, 0, z), (2, 2, 2), (255, 255, 255))]))
        rr.render()
        widths.append(int(rr.get_framebuffer().any(axis=(0, 2)).sum()))
    assert abs(widths[0] - widths[1]) <= 1 and widths[0] > 10

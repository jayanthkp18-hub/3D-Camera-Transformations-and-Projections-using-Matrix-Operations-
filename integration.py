"""
Glue between the teammates' modules and Person 3's renderer.

This is CLIENT code: it only talks to the stable `renderer` API and to the
other persons' files (math_module.py, camera.py, scene_cube.py,
scene_pyramid.py). The renderer package itself never imports them.

Why this file exists
--------------------
The renderer takes 4x4 matrices, but the repo currently has no 4x4 view or
projection matrix yet:
  * Person 1's math_module projects point by point (x*f/z) with a 3x3
    rotation from look_at_rotation().
  * Person 2's Camera class has no get_view_matrix() yet.
So `view_matrix()` and `perspective_matrix()` below are bridges that give the
SAME result as the team's math, expressed as matrices. When Persons 1 / 2
publish their own 4x4 matrices, swap them in here; nothing else changes.

Team convention (derived from math_module.py)
---------------------------------------------
camera space: x right, y up, z FORWARD (camera looks along +z), left-handed.
"""

import numpy as np

import math_module                       # Person 1
from renderer import (Mesh, Scene, BOX_TRIANGLES, BOX_EDGES,
                      PYRAMID_TRIANGLES)

import scene_cube                        # Person 4
import scene_pyramid                     # Person 4


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------

def view_matrix(camera_position, camera_target):
    """
    4x4 view matrix equal to Person 1's  P_cam = R^T (P - C)  with
    R = math_module.look_at_rotation(C, target).
    """
    C = np.asarray(camera_position, dtype=float)
    R = math_module.look_at_rotation(C, np.asarray(camera_target, dtype=float))
    V = np.eye(4)
    V[:3, :3] = R.T
    V[:3, 3] = -R.T @ C
    return V


def view_matrix_from_camera(camera):
    """
    Accepts Person 2's Camera. Uses camera.get_view_matrix() once it exists,
    otherwise falls back to position + target (look-at).
    NOTE: Camera's pitch/yaw/roll are not used yet because camera.py does not
    produce a view matrix from them.
    """
    if hasattr(camera, "get_view_matrix"):
        return np.asarray(camera.get_view_matrix(), dtype=float)
    return view_matrix(camera.get_position(), camera.get_target())


def perspective_matrix(fov_y_deg, aspect, near, far):
    """
    Perspective projection for a camera looking along +z.
    clip.w = z_cam, so x_ndc = (f / aspect) * x / z and y_ndc = f * y / z
    with f = 1 / tan(fov/2): exactly Person 1's  focal * x / z,  rescaled.
    """
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    P = np.zeros((4, 4))
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = 1.0
    return P


def orthographic_matrix(left, right, bottom, top, near, far):
    """Orthographic projection for a camera looking along +z (w = 1)."""
    P = np.eye(4)
    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = 2.0 / (far - near)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(far + near) / (far - near)
    return P


def projection_from_scene_module(mod=scene_cube, prefix=""):
    """
    Build the projection matrix from the settings Person 4 stores in a scene
    module (projection_type, field_of_view, near/far, ortho bounds, screen
    size). Use prefix="pyramid_" for scene_pyramid.
    """
    g = lambda name: getattr(mod, prefix + name)
    w, h = g("screen_width"), g("screen_height")
    if g("projection_type") == "orthographic":
        return orthographic_matrix(g("ortho_left"), g("ortho_right"),
                                   g("ortho_bottom"), g("ortho_top"),
                                   g("near_plane"), g("far_plane"))
    return perspective_matrix(g("field_of_view"), w / h,
                              g("near_plane"), g("far_plane"))


def focal_length_pixels(fov_y_deg, height):
    """Focal length (in pixels) that matches a vertical field of view."""
    return (height / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)


# --------------------------------------------------------------------------
# Scene data from Person 4
# --------------------------------------------------------------------------
# NOTE: Person 4's vertices are already in world coordinates (the pyramid's
# x range is 2..4 and pyramid_position is also [3, 0, 0], so applying the
# position again would double-shift it). Vertices are used as they are;
# *_position / *_rotation / *_scale are identity or already baked in.

def cube_mesh(face_color=(70, 130, 220), edge_color=(255, 255, 255),
              solid=True, wire=True):
    return Mesh(scene_cube.vertices,
                BOX_TRIANGLES.copy() if solid else None, face_color,
                np.array(scene_cube.edges) if wire else None, edge_color)


def pyramid_mesh(face_color=(220, 120, 60), edge_color=(255, 255, 255),
                 solid=True, wire=True):
    return Mesh(scene_pyramid.pyramid_vertices,
                PYRAMID_TRIANGLES.copy() if solid else None, face_color,
                np.array(scene_pyramid.pyramid_edges) if wire else None,
                edge_color)


def build_team_scene(solid=True, wire=True):
    return Scene([cube_mesh(solid=solid, wire=wire),
                  pyramid_mesh(solid=solid, wire=wire)])

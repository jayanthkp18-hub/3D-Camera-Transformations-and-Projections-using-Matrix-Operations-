"""
Glue between the teammates' modules and Person 3's renderer.

This file connects the mathematical modules, camera, scene data,
transformation matrices and the renderer.

Transformation pipeline:
Model -> View -> Projection -> Screen
"""

import numpy as np

import math_module  # Person 1
import scene_cube  # Person 4
import scene_pyramid  # Person 4
from renderer import (
    BOX_EDGES,
    BOX_TRIANGLES,
    PYRAMID_TRIANGLES,
    Mesh,
    Scene,
    create_renderer,
)


# Create the 4x4 view matrix from camera position and target
def view_matrix(camera_position, camera_target):
    C = np.asarray(camera_position, dtype=float)

    R = math_module.look_at_rotation(
        C,
        np.asarray(camera_target, dtype=float)
    )

    V = np.eye(4)

    # Apply inverse camera rotation
    V[:3, :3] = R.T

    # Apply camera translation
    V[:3, 3] = -R.T @ C

    return V


# Create view matrix directly from Person 2's Camera class
def view_matrix_from_camera(camera):
    if hasattr(camera, "get_view_matrix"):
        return np.asarray(
            camera.get_view_matrix(),
            dtype=float
        )

    return view_matrix(
        camera.get_position(),
        camera.get_target()
    )


# Create the 4x4 perspective projection matrix
def perspective_matrix(fov_y_deg, aspect, near, far):
    f = 1.0 / np.tan(
        np.radians(fov_y_deg) / 2.0
    )

    P = np.zeros((4, 4))

    P[0, 0] = f / aspect
    P[1, 1] = f

    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)

    # Perspective division uses z as w
    P[3, 2] = 1.0

    return P


# Create the 4x4 orthographic projection matrix
def orthographic_matrix(left, right, bottom, top, near, far):
    P = np.eye(4)

    P[0, 0] = 2.0 / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[2, 2] = 2.0 / (far - near)

    P[0, 3] = -(right + left) / (right - left)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 3] = -(far + near) / (far - near)

    return P


# Create projection matrix using values stored in the scene module
def projection_from_scene_module(mod=scene_cube, prefix=""):
    g = lambda name: getattr(mod, prefix + name)

    width = g("screen_width")
    height = g("screen_height")

    if g("projection_type") == "orthographic":
        return orthographic_matrix(
            g("ortho_left"),
            g("ortho_right"),
            g("ortho_bottom"),
            g("ortho_top"),
            g("near_plane"),
            g("far_plane")
        )

    return perspective_matrix(
        g("field_of_view"),
        width / height,
        g("near_plane"),
        g("far_plane")
    )


# Calculate focal length in pixels from field of view
def focal_length_pixels(fov_y_deg, height):
    return (
        (height / 2.0) /
        np.tan(np.radians(fov_y_deg) / 2.0)
    )


# Create cube mesh using Person 4's scene data
def cube_mesh(
    face_color=(70, 130, 220),
    edge_color=(255, 255, 255),
    solid=True,
    wire=True
):
    return Mesh(
        scene_cube.vertices,
        BOX_TRIANGLES.copy() if solid else None,
        face_color,
        np.array(scene_cube.edges) if wire else None,
        edge_color
    )


# Create pyramid mesh using Person 4's scene data
def pyramid_mesh(
    face_color=(220, 120, 60),
    edge_color=(255, 255, 255),
    solid=True,
    wire=True
):
    return Mesh(
        scene_pyramid.pyramid_vertices,
        PYRAMID_TRIANGLES.copy() if solid else None,
        face_color,
        np.array(scene_pyramid.pyramid_edges) if wire else None,
        edge_color
    )


# Build the scene according to the selected object
def build_team_scene(object_choice="Both", solid=True, wire=True):
    objects = []

    if object_choice == "Cube":
        objects.append(
            cube_mesh(
                solid=solid,
                wire=wire
            )
        )

    elif object_choice == "Pyramid":
        objects.append(
            pyramid_mesh(
                solid=solid,
                wire=wire
            )
        )

    else:
        objects.append(
            cube_mesh(
                solid=solid,
                wire=wire
            )
        )

        objects.append(
            pyramid_mesh(
                solid=solid,
                wire=wire
            )
        )

    return Scene(objects)


# Render the selected object using model, view and projection matrices
def render_scene(
    object_choice="Both",
    camera_position=None,
    camera_target=None,
    projection_matrix=None,
    model_matrix=None
):
    # Create renderer using the scene dimensions
    r = create_renderer(
        scene_cube.screen_width,
        scene_cube.screen_height
    )

    # Use default camera values if none are supplied
    if camera_position is None:
        camera_position = scene_cube.camera_position

    if camera_target is None:
        camera_target = scene_cube.camera_target

    # Calculate view matrix from current camera position and target
    V = view_matrix(
        camera_position,
        camera_target
    )

    # Use the projection matrix supplied by app.py
    if projection_matrix is None:
        P = projection_from_scene_module(scene_cube)
    else:
        P = projection_matrix

    # Identity matrix means no model transformation
    if model_matrix is None:
        model_matrix = np.eye(4)

    # Apply model transformation to object vertices
    def transform_vertices(vertices):
        vertices = np.asarray(
            vertices,
            dtype=float
        )

        # Convert vertices to homogeneous coordinates
        homogeneous = np.hstack([
            vertices,
            np.ones((len(vertices), 1))
        ])

        # Apply model matrix
        transformed = (
            model_matrix @ homogeneous.T
        ).T

        # Convert back to 3D coordinates
        return transformed[:, :3]

    # Transform cube vertices using the model matrix
    cube_vertices = transform_vertices(
        scene_cube.vertices
    )

    cube = Mesh(
        cube_vertices,
        BOX_TRIANGLES.copy(),
        (70, 130, 220),
        np.array(scene_cube.edges),
        (255, 255, 255)
    )

    # Transform pyramid vertices using the model matrix
    pyramid_vertices = transform_vertices(
        scene_pyramid.pyramid_vertices
    )

    pyramid = Mesh(
        pyramid_vertices,
        PYRAMID_TRIANGLES.copy(),
        (220, 120, 60),
        np.array(scene_pyramid.pyramid_edges),
        (255, 255, 255)
    )

    # Select which object should be rendered
    if object_choice == "Cube":
        scene = Scene([
            cube
        ])

    elif object_choice == "Pyramid":
        scene = Scene([
            pyramid
        ])

    else:
        scene = Scene([
            cube,
            pyramid
        ])

    # Pass view and projection matrices to the renderer
    r.set_view_matrix(V)
    r.set_projection_matrix(P)

    # Set the selected scene
    r.set_scene(scene)

    # Set the background color
    r.set_clear_color(
        (20, 22, 30)
    )

    # Render the scene
    r.render()

    # Return the rendered image
    return r.get_framebuffer()
import numpy as np

pyramid_vertices = np.array([
    [2.0, -1.0, -1.0],
    [4.0, -1.0, -1.0],
    [4.0, -1.0,  1.0],
    [2.0, -1.0,  1.0],
    [3.0,  2.0,  0.0]
], dtype=float)

pyramid_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4)
]

pyramid_position = np.array([3.0, 0.0, 0.0])
pyramid_rotation = np.array([0.0, 0.0, 0.0])
pyramid_scale = np.array([1.0, 1.0, 1.0])

pyramid_camera_position = np.array([5.0, 5.0, 8.0])
pyramid_camera_target = np.array([0.0, 0.0, 0.0])
pyramid_camera_rotation = np.array([0.0, 0.0, 0.0])

pyramid_projection_type = "perspective"

pyramid_focal_length = 50.0
pyramid_field_of_view = 60.0

pyramid_near_plane = 0.1
pyramid_far_plane = 100.0

pyramid_ortho_left = -5.0
pyramid_ortho_right = 5.0
pyramid_ortho_bottom = -5.0
pyramid_ortho_top = 5.0

pyramid_screen_width = 1000
pyramid_screen_height = 700
import numpy as np

vertices = np.array([
    [-1.0, -1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [ 1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [ 1.0,  1.0,  1.0],
    [-1.0,  1.0,  1.0]
], dtype=float)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

object_position = np.array([0.0, 0.0, 0.0])
object_rotation = np.array([0.0, 0.0, 0.0])
object_scale = np.array([1.0, 1.0, 1.0])

camera_position = np.array([5.0, 5.0, 8.0])
camera_target = np.array([0.0, 0.0, 0.0])

camera_rotation = np.array([0.0, 0.0, 0.0])

projection_type = "perspective"

focal_length = 50.0
field_of_view = 60.0

near_plane = 0.1
far_plane = 100.0

ortho_left = -5.0
ortho_right = 5.0
ortho_bottom = -5.0
ortho_top = 5.0

screen_width = 1000
screen_height = 700
import numpy as np

from math_module import (
    RotateX,
    RotateY,
    RotateZ,
    world_to_camera
)


# ============================================================
# CAMERA CLASS
# Camera position + orientation + camera control
# ============================================================

class Camera:

    def __init__(self, position=None, target=None):

        # ----------------------------------------------------
        # Camera position in world coordinates
        # ----------------------------------------------------

        if position is None:
            self.position = np.array(
                [0.0, 0.0, -5.0],
                dtype=float
            )
        else:
            self.position = np.array(
                position,
                dtype=float
            )

        # ----------------------------------------------------
        # Point the camera is looking at
        # ----------------------------------------------------

        if target is None:
            self.target = np.array(
                [0.0, 0.0, 0.0],
                dtype=float
            )
        else:
            self.target = np.array(
                target,
                dtype=float
            )

        # ----------------------------------------------------
        # Camera orientation
        #
        # pitch = rotation around X-axis
        # yaw   = rotation around Y-axis
        # roll  = rotation around Z-axis
        # ----------------------------------------------------

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    # ========================================================
    # CAMERA POSITION
    # ========================================================

    def set_position(self, x, y, z):
        """Set camera position in world coordinates."""

        self.position = np.array(
            [x, y, z],
            dtype=float
        )

    def move(self, dx, dy, dz):
        """Move camera relative to its current position."""

        self.position += np.array(
            [dx, dy, dz],
            dtype=float
        )

    def get_position(self):
        """Return current camera position."""

        return self.position.copy()

    # ========================================================
    # CAMERA TARGET
    # ========================================================

    def set_target(self, x, y, z):
        """Set the point the camera is looking at."""

        self.target = np.array(
            [x, y, z],
            dtype=float
        )

    def get_target(self):
        """Return current camera target."""

        return self.target.copy()

    # ========================================================
    # CAMERA ROTATION
    # ========================================================

    def set_rotation(self, pitch, yaw, roll):
        """
        Set camera orientation.

        pitch -> X-axis
        yaw   -> Y-axis
        roll  -> Z-axis
        """

        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

    def rotate(self, dpitch, dyaw, droll):
        """Rotate camera relative to its current orientation."""

        self.pitch += dpitch
        self.yaw += dyaw
        self.roll += droll

    def get_rotation(self):
        """Return pitch, yaw and roll."""

        return (
            self.pitch,
            self.yaw,
            self.roll
        )

    # ========================================================
    # CAMERA ROTATION MATRIX
    # ========================================================

    def get_rotation_matrix(self):
        """
        Create the camera rotation matrix.

        Uses the rotation functions from math_module.py.
        """

        Rx = RotateX(
            np.radians(self.pitch)
        )

        Ry = RotateY(
            np.radians(self.yaw)
        )

        Rz = RotateZ(
            np.radians(self.roll)
        )

        # Rotation order:
        #
        # R = Rz @ Ry @ Rx

        R = Rz @ Ry @ Rx

        return R

    # ========================================================
    # CAMERA DIRECTION
    # ========================================================

    def get_forward_direction(self):
        """
        Return normalized direction from camera
        towards its target.
        """

        direction = self.target - self.position

        length = np.linalg.norm(direction)

        if length == 0:
            raise ValueError(
                "Camera position and target cannot be the same."
            )

        return direction / length

    # ========================================================
    # LOOK AT
    # ========================================================

    def look_at(self, x, y, z):
        """Make the camera look at a specific point."""

        self.target = np.array(
            [x, y, z],
            dtype=float
        )

    # ========================================================
    # WORLD -> CAMERA
    # ========================================================

    def world_to_camera(self, point):
        """
        Convert a world-space point to camera-space.

        Uses Person 1's world_to_camera() function
        from math_module.py.
        """

        R = self.get_rotation_matrix()

        return world_to_camera(
            np.asarray(point, dtype=float),
            self.position,
            R
        )

    # ========================================================
    # CAMERA STATE
    # ========================================================

    def get_state(self):
        """Return the current camera state."""

        return {
            "position": self.get_position(),
            "target": self.get_target(),
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """Reset camera to default state."""

        self.position = np.array(
            [0.0, 0.0, -5.0],
            dtype=float
        )

        self.target = np.array(
            [0.0, 0.0, 0.0],
            dtype=float
        )

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0


# ============================================================
# TEST CAMERA
# ============================================================

if __name__ == "__main__":

    camera = Camera()

    # Set camera position
    camera.set_position(0, 0, -5)

    # Set camera target
    camera.set_target(0, 0, 0)

    # Set camera rotation
    camera.set_rotation(
        pitch=0,
        yaw=0,
        roll=0
    )

    print("Camera Position:")
    print(camera.get_position())

    print("\nCamera Target:")
    print(camera.get_target())

    print("\nCamera Rotation:")
    print(camera.get_rotation())

    print("\nForward Direction:")
    print(camera.get_forward_direction())

    print("\nRotation Matrix:")
    print(camera.get_rotation_matrix())

    # Test world -> camera transformation
    point = np.array([2.0, 3.0, 4.0])

    print("\nWorld Point:")
    print(point)

    print("\nCamera Point:")
    print(camera.world_to_camera(point))
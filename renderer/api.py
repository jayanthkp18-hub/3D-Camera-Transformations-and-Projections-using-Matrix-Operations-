"""
Stable public API for Person 3 (Graphics Engine / Renderer).

THIS FILE IS THE CONTRACT. Client code (Persons 1, 2, 4 and the main program)
must only ever talk to `Renderer`. Any backend (Python today, C++ later) has
to implement exactly these methods with exactly these data formats.

Conventions (agree on these with Persons 1 and 2!)
--------------------------------------------------
* Matrices are 4x4, ROW-MAJOR numpy arrays (dtype float32/float64 accepted,
  converted internally to contiguous float32), used with COLUMN vectors:
        clip = Projection @ View @ [x, y, z, 1]^T
* The renderer only assumes CLIP SPACE: after dividing by w, x and y are in
  [-1, 1] (x right, y up) and z is in [-1, 1] with -1 = near plane and
  +1 = far plane; w > 0 means "in front of the camera".
  It does not care which way the camera looks. The team's camera
  (math_module.look_at_rotation) looks down +Z in a left-handed space, and
  integration.py builds matrices for exactly that.
* Framebuffer is (height, width, 3) uint8 RGB, row 0 = TOP of the image.
* Scene geometry is already in WORLD coordinates (Person 4 bakes
  position/size into the vertices, see scene.py helpers).
* A triangle is front-facing when it appears counter-clockwise on screen
  (only matters when cull_backfaces is on).
"""

from abc import ABC, abstractmethod

import numpy as np


class Renderer(ABC):
    """Abstract renderer. Implemented by every backend."""

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def __init__(self, width: int, height: int): ...

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Change output resolution (clears buffers)."""

    # ---- inputs from other team members ---------------------------------
    @abstractmethod
    def set_view_matrix(self, matrix: np.ndarray) -> None:
        """4x4 view matrix from Person 2."""

    @abstractmethod
    def set_projection_matrix(self, matrix: np.ndarray) -> None:
        """4x4 projection matrix from Person 1."""

    @abstractmethod
    def set_scene(self, scene) -> None:
        """Scene (see scene.py) from Person 4."""

    # ---- optional settings (sensible defaults, safe to ignore) ----------
    @abstractmethod
    def set_clear_color(self, rgb) -> None:
        """Background colour, three ints 0..255."""

    @abstractmethod
    def set_options(self, **options) -> None:
        """
        Tweak behaviour. Known keys (unknown keys must be ignored):
          cull_backfaces (bool, default False)
          lighting       (bool, default True)  simple directional light
          light_dir      (3 floats, world space, direction light travels)
          ambient        (float 0..1, default 0.35)
          wireframe      (bool, default False) draw triangle outlines only
          line_width     (int, default 1) pixel width of edges / outlines
        """

    # ---- actions ---------------------------------------------------------
    @abstractmethod
    def render(self) -> None:
        """Run the full pipeline and fill the framebuffer + depth buffer."""

    @abstractmethod
    def get_framebuffer(self) -> np.ndarray:
        """(height, width, 3) uint8 RGB image, row 0 at the top."""

    @abstractmethod
    def get_depthbuffer(self) -> np.ndarray:
        """(height, width) float32 NDC depth in [-1, 1]; +inf = empty."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Counters from the last render(): triangles_in, triangles_drawn, ..."""

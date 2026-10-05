"""
Person 3 - Graphics Engine / Renderer.

Client code should ONLY do:

    from renderer import create_renderer, Scene, make_box, make_ground
    r = create_renderer(640, 480)          # backend chosen automatically
    r.set_view_matrix(V); r.set_projection_matrix(P); r.set_scene(scene)
    r.render(); image = r.get_framebuffer()

Swapping Python for C++ later
-----------------------------
1. Build a C++ extension module named `_renderer_cpp` (pybind11) that exposes
   a class `CppRenderer` with the same methods as `api.Renderer`
   (numpy arrays in / numpy arrays out).
2. Drop it next to this package. `create_renderer()` will pick it up
   automatically (backend="auto"), or force it with backend="cpp".
No client code changes are needed, because clients never import a backend.
"""

import os

from .api import Renderer
from .scene import (Scene, Mesh, make_box, make_ground, make_pyramid,
                    BOX_TRIANGLES, BOX_EDGES, PYRAMID_TRIANGLES)
from .python_backend import PythonRenderer

__all__ = ["Renderer", "Scene", "Mesh", "make_box", "make_ground",
           "make_pyramid", "BOX_TRIANGLES", "BOX_EDGES", "PYRAMID_TRIANGLES",
           "create_renderer", "available_backends"]


def _load_cpp():
    try:
        from _renderer_cpp import CppRenderer      # compiled pybind11 module
        return CppRenderer
    except ImportError:
        return None


def available_backends():
    names = ["python"]
    if _load_cpp() is not None:
        names.append("cpp")
    return names


def create_renderer(width: int, height: int, backend: str = "auto") -> Renderer:
    """
    backend: "auto" (cpp if available, else python), "python" or "cpp".
    Can also be overridden with the environment variable RENDERER_BACKEND.
    """
    backend = os.environ.get("RENDERER_BACKEND", backend).lower()
    if backend in ("auto", "cpp"):
        cls = _load_cpp()
        if cls is not None:
            return cls(width, height)
        if backend == "cpp":
            raise RuntimeError("C++ backend requested but _renderer_cpp "
                               "is not installed")
    if backend not in ("auto", "python", "cpp"):
        raise ValueError(f"unknown backend {backend!r}")
    return PythonRenderer(width, height)

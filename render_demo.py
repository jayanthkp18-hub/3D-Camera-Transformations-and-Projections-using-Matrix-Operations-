"""
Renders the team's cube + pyramid with the real camera / scene settings.

Run from the repository root:   python render_demo.py
Writes PNG files next to this script.
"""

import numpy as np

import scene_cube as cfg                          # Person 4's settings
from renderer import create_renderer
from renderer.imageio import save_png
from integration import (build_team_scene, view_matrix,
                         projection_from_scene_module, orthographic_matrix)


def main():
    r = create_renderer(cfg.screen_width, cfg.screen_height)
    r.set_view_matrix(view_matrix(cfg.camera_position, cfg.camera_target))
    r.set_projection_matrix(projection_from_scene_module(cfg))
    r.set_clear_color((20, 22, 30))

    # 1. solid faces with white edges on top
    r.set_scene(build_team_scene(solid=True, wire=True))
    r.set_options(line_width=2)
    r.render()
    save_png("render_solid.png", r.get_framebuffer())
    print("solid     ", r.get_stats())

    # 2. pure wireframe (exactly what Person 4's vertices + edges describe)
    r.set_scene(build_team_scene(solid=False, wire=True))
    r.render()
    save_png("render_wireframe.png", r.get_framebuffer())
    print("wireframe ", r.get_stats())

    # 3. same scene, orthographic projection (no perspective shrinking)
    r.set_scene(build_team_scene(solid=True, wire=True))
    r.set_projection_matrix(orthographic_matrix(
        cfg.ortho_left, cfg.ortho_right, cfg.ortho_bottom, cfg.ortho_top,
        cfg.near_plane, cfg.far_plane))
    r.render()
    save_png("render_orthographic.png", r.get_framebuffer())
    print("ortho     ", r.get_stats())


if __name__ == "__main__":
    main()

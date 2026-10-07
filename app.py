import numpy as np
import streamlit as st

import scene_cube
import scene_pyramid
from integration import (
    orthographic_matrix,
    perspective_matrix,
    render_scene,
    view_matrix,
)


# Display a NumPy matrix as a proper mathematical matrix
def display_matrix(matrix):
    rows = []

    for row in matrix:
        values = [f"{value:.4f}" for value in row]
        rows.append(" & ".join(values))

    matrix_latex = (
        r"\begin{bmatrix}"
        + r" \\ ".join(rows)
        + r"\end{bmatrix}"
    )

    st.latex(matrix_latex)


# Display a vector as a mathematical column vector
def display_vector(vector):
    rows = []

    for value in vector:
        rows.append(f"{value:.4f}")

    vector_latex = (
        r"\begin{bmatrix}"
        + r" \\ ".join(rows)
        + r"\end{bmatrix}"
    )

    st.latex(vector_latex)


# Create a 4x4 rotation matrix from X, Y and Z rotation angles
def rotation_matrix(rx, ry, rz):
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return Rz @ Ry @ Rx


# Create a 4x4 model matrix using rotation, translation and scale
def model_matrix(rx, ry, rz, tx, ty, tz, scale):
    M = rotation_matrix(rx, ry, rz)

    M[:3, :3] *= scale

    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz

    return M


# Page configuration
st.set_page_config(
    page_title="3D Camera Transformations",
    layout="wide"
)

st.title("3D Camera Transformations and Projection")

st.caption(
    "Interactive visualization of model, view and projection "
    "transformations using 3D objects."
)


# Object selection
st.subheader("Object Selection")

object_choice = st.radio(
    "Select Object",
    ["Cube", "Pyramid", "Both"],
    horizontal=True
)


# Projection selection
st.subheader("Projection Type")

projection_choice = st.radio(
    "Select Projection",
    ["Perspective", "Orthographic"],
    horizontal=True
)


# Camera controls
st.subheader("Camera Controls")

col1, col2, col3 = st.columns(3)

with col1:
    camera_x = st.slider(
        "Camera X",
        -20.0,
        20.0,
        float(scene_cube.camera_position[0]),
        0.5
    )

with col2:
    camera_y = st.slider(
        "Camera Y",
        -20.0,
        20.0,
        float(scene_cube.camera_position[1]),
        0.5
    )

with col3:
    camera_z = st.slider(
        "Camera Z",
        1.0,
        30.0,
        float(scene_cube.camera_position[2]),
        0.5
    )

col1, col2, col3 = st.columns(3)

with col1:
    target_x = st.slider(
        "Target X",
        -10.0,
        10.0,
        float(scene_cube.camera_target[0]),
        0.5
    )

with col2:
    target_y = st.slider(
        "Target Y",
        -10.0,
        10.0,
        float(scene_cube.camera_target[1]),
        0.5
    )

with col3:
    target_z = st.slider(
        "Target Z",
        -10.0,
        20.0,
        float(scene_cube.camera_target[2]),
        0.5
    )

camera_position = np.array([
    camera_x,
    camera_y,
    camera_z
])

camera_target = np.array([
    target_x,
    target_y,
    target_z
])


# Model transformation controls
st.subheader("Object Transformations")

col1, col2, col3 = st.columns(3)

with col1:
    rotation_x = st.slider(
        "Rotation X",
        -180.0,
        180.0,
        0.0,
        5.0
    )

with col2:
    rotation_y = st.slider(
        "Rotation Y",
        -180.0,
        180.0,
        0.0,
        5.0
    )

with col3:
    rotation_z = st.slider(
        "Rotation Z",
        -180.0,
        180.0,
        0.0,
        5.0
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    translation_x = st.slider(
        "Translation X",
        -10.0,
        10.0,
        0.0,
        0.5
    )

with col2:
    translation_y = st.slider(
        "Translation Y",
        -10.0,
        10.0,
        0.0,
        0.5
    )

with col3:
    translation_z = st.slider(
        "Translation Z",
        -10.0,
        10.0,
        0.0,
        0.5
    )

with col4:
    scale = st.slider(
        "Scale",
        0.1,
        3.0,
        1.0,
        0.1
    )

M = model_matrix(
    rotation_x,
    rotation_y,
    rotation_z,
    translation_x,
    translation_y,
    translation_z,
    scale
)


# Projection controls
st.subheader("Projection Controls")

if projection_choice == "Perspective":

    col1, col2, col3 = st.columns(3)

    with col1:
        fov = st.slider(
            "Field of View",
            30.0,
            120.0,
            float(scene_cube.field_of_view),
            1.0
        )

    with col2:
        near_plane = st.slider(
            "Near Plane",
            0.1,
            10.0,
            float(scene_cube.near_plane),
            0.1
        )

    with col3:
        far_plane = st.slider(
            "Far Plane",
            20.0,
            200.0,
            float(scene_cube.far_plane),
            5.0
        )

else:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ortho_left = st.slider(
            "Left",
            -10.0,
            0.0,
            float(scene_cube.ortho_left),
            0.5
        )

    with col2:
        ortho_right = st.slider(
            "Right",
            0.0,
            10.0,
            float(scene_cube.ortho_right),
            0.5
        )

    with col3:
        ortho_bottom = st.slider(
            "Bottom",
            -10.0,
            0.0,
            float(scene_cube.ortho_bottom),
            0.5
        )

    with col4:
        ortho_top = st.slider(
            "Top",
            0.0,
            10.0,
            float(scene_cube.ortho_top),
            0.5
        )

    col1, col2 = st.columns(2)

    with col1:
        near_plane = st.slider(
            "Near Plane",
            0.1,
            10.0,
            float(scene_cube.near_plane),
            0.1
        )

    with col2:
        far_plane = st.slider(
            "Far Plane",
            20.0,
            200.0,
            float(scene_cube.far_plane),
            5.0
        )


# Calculate View Matrix
V = view_matrix(
    camera_position,
    camera_target
)


# Calculate Projection Matrix
if projection_choice == "Perspective":

    aspect = (
        scene_cube.screen_width /
        scene_cube.screen_height
    )

    P = perspective_matrix(
        fov,
        aspect,
        near_plane,
        far_plane
    )

else:

    P = orthographic_matrix(
        ortho_left,
        ortho_right,
        ortho_bottom,
        ortho_top,
        near_plane,
        far_plane
    )


# Render the selected object
image = render_scene(
    object_choice=object_choice,
    camera_position=camera_position,
    camera_target=camera_target,
    projection_matrix=P,
    model_matrix=M
)


# Display rendered scene
st.subheader("3D Scene")

st.image(
    image,
    caption=f"{object_choice} - {projection_choice} Projection",
    use_container_width=True
)


# Camera information
st.subheader("Camera Information")

col1, col2 = st.columns(2)

with col1:
    st.write(
        f"**Camera Position:** "
        f"({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})"
    )

with col2:
    st.write(
        f"**Camera Target:** "
        f"({target_x:.2f}, {target_y:.2f}, {target_z:.2f})"
    )


# Object transformation information
st.subheader("Object Transformation")

st.write(
    f"**Rotation:** "
    f"X = {rotation_x:.1f}°, "
    f"Y = {rotation_y:.1f}°, "
    f"Z = {rotation_z:.1f}°"
)

st.write(
    f"**Translation:** "
    f"({translation_x:.2f}, "
    f"{translation_y:.2f}, "
    f"{translation_z:.2f})"
)

st.write(
    f"**Scale:** {scale:.2f}"
)


# Projection information
st.subheader("Projection Information")

st.write(
    f"**Projection Type:** {projection_choice}"
)

if projection_choice == "Perspective":

    st.write(
        f"**Field of View:** {fov:.2f}°"
    )

    st.write(
        f"**Near Plane:** {near_plane:.2f}"
    )

    st.write(
        f"**Far Plane:** {far_plane:.2f}"
    )

else:

    st.write(
        f"**Left:** {ortho_left:.2f}"
    )

    st.write(
        f"**Right:** {ortho_right:.2f}"
    )

    st.write(
        f"**Bottom:** {ortho_bottom:.2f}"
    )

    st.write(
        f"**Top:** {ortho_top:.2f}"
    )

    st.write(
        f"**Near Plane:** {near_plane:.2f}"
    )

    st.write(
        f"**Far Plane:** {far_plane:.2f}"
    )


# Transformation matrices
st.subheader("Transformation Matrices")

with st.expander("Model Matrix", expanded=True):
    display_matrix(M)

with st.expander("View Matrix", expanded=True):
    display_matrix(V)

with st.expander("Projection Matrix", expanded=True):
    display_matrix(P)


# Transformation Pipeline
st.subheader("Transformation Pipeline")

st.markdown("""
**Object Coordinates**
↓
**Model Transformation**
↓
**World Coordinates**
↓
**View Transformation**
↓
**Camera Coordinates**
↓
**Projection Transformation**
↓
**Clip Coordinates**
↓
**Perspective Division / NDC**
↓
**Screen Coordinates**
""")


# Matrix Multiplication Demonstration
with st.expander(
    "Matrix Multiplication Demonstration",
    expanded=True
):

    demo_object = st.selectbox(
        "Select Object",
        ["Cube", "Pyramid"]
    )

    if demo_object == "Cube":
        demo_vertices = np.asarray(
            scene_cube.vertices,
            dtype=float
        )
    else:
        demo_vertices = np.asarray(
            scene_pyramid.pyramid_vertices,
            dtype=float
        )

    vertex_index = st.number_input(
        "Select Vertex Index",
        min_value=0,
        max_value=len(demo_vertices) - 1,
        value=0,
        step=1
    )

    selected_vertex = demo_vertices[int(vertex_index)]
    vertex_h = np.append(selected_vertex, 1.0)

    # Apply transformations
    world_vertex = M @ vertex_h
    camera_vertex = V @ world_vertex
    clip_vertex = P @ camera_vertex

    # Perspective division
    if abs(clip_vertex[3]) > 1e-8:
        ndc_vertex = clip_vertex[:3] / clip_vertex[3]
    else:
        ndc_vertex = np.full(3, np.nan)

    # Convert NDC to screen coordinates
    screen_width = scene_cube.screen_width
    screen_height = scene_cube.screen_height

    screen_x = (
        (ndc_vertex[0] + 1) *
        screen_width /
        2
    )

    screen_y = (
        (1 - ndc_vertex[1]) *
        screen_height /
        2
    )

    # Selected vertex
    st.write("### 1. Selected Vertex")
    display_vector(vertex_h)

    # Model transformation
    st.write("### 2. Model Transformation")
    st.write("**M × Vertex = World Coordinates**")

    display_matrix(M)
    display_vector(vertex_h)

    st.write("**= World Coordinates**")
    display_vector(world_vertex)

    # View transformation
    st.write("### 3. View Transformation")
    st.write("**V × World Coordinates = Camera Coordinates**")

    display_matrix(V)
    display_vector(world_vertex)

    st.write("**= Camera Coordinates**")
    display_vector(camera_vertex)

    # Projection transformation
    st.write("### 4. Projection Transformation")
    st.write("**P × Camera Coordinates = Clip Coordinates**")

    display_matrix(P)
    display_vector(camera_vertex)

    st.write("**= Clip Coordinates**")
    display_vector(clip_vertex)

    # NDC
    st.write("### 5. Perspective Division / NDC")
    st.write("**NDC = Clip Coordinates / w**")
    display_vector(ndc_vertex)

    # Screen coordinates
    st.write("### 6. Screen Coordinates")

    st.write(
        f"**Screen X:** {screen_x:.2f} px"
    )

    st.write(
        f"**Screen Y:** {screen_y:.2f} px"
    )

    st.write(
        f"**Screen Position:** "
        f"({screen_x:.2f}, {screen_y:.2f}) px"
    )

    # Combined transformation
    st.write("### 7. Combined Transformation")

    PVM = P @ V @ M

    st.write("**P × V × M**")
    display_matrix(PVM)

    combined_clip = PVM @ vertex_h

    st.write(
        "**(P × V × M) × Vertex = Clip Coordinates**"
    )

    display_vector(combined_clip)

    # Verification
    st.write("### 8. Verification")

    step_by_step = P @ V @ M @ vertex_h
    combined_result = PVM @ vertex_h

    difference = np.max(
        np.abs(step_by_step - combined_result)
    )

    st.write(
        f"Maximum difference: **{difference:.8f}**"
    )

    if difference < 1e-6:
        st.success(
            "Both transformation methods produce the same result."
        )
    else:
        st.warning(
            "The transformation results are different."
        )
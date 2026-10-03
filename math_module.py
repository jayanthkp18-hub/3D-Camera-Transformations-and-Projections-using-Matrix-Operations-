import numpy as np



# ============================================================
# rotation, translation and M part ^_~
# ============================================================

def RotateX(theta):
    c= np.cos(theta)
    s= np.sin(theta)
    
    return np.array([[1,0,0],
                    [0,c,-s],
                    [0,s,c]])
    
def RotateY(theta):
    c= np.cos(theta)
    s= np.sin(theta)
    
    return np.array([[c,0,s],
                     [0,1,0],
                     [-s,0,c]])

def RotateZ(theta):
    c= np.cos(theta)
    s= np.sin(theta)
    
    return np.array([[c,-s,0],
                     [s,c,0],
                     [0,0,1]])

def translation_matrix(tx, ty, tz):
    
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def transformation_matrix(R, tx, ty, tz):
    M = np.eye(4)

    M[:3, :3] = R                   #R is respective rotation matrix
    M[:3, 3] = [tx, ty, tz]         # look at the above one, trasnlation matrix this thingy is

    return M

def transform_point(M, P):
    
    return M @ P        # this boi is combo of translation and rotation, i.e pos after rotation and translation



# ============================================================
# calculatio related function to ez the work boiiii part ^_~
# ============================================================


def homogeneous_point(x, y, z):
    
    return np.array([x, y, z, 1])


def world_to_camera(P,C,R):
    relative = P - C
    
    return R.T @ relative           #this thingy is P_cam=Rt*(P-C)


def is_orthogonal(R):
    I = np.eye(3)
    
    # checking for this boi --> Rt.R = I
    # allclose checks if all elements of the two arrays are equal within a tolerance
    return np.allclose(R.T @ R, I)



# ============================================================
# All projection mathematics part ^_~
# ============================================================


def perspective_projection(P, focal_length):
    x, y, z = P

    if z == 0:
        raise ValueError("Cannot project a point with z = 0")

    return np.array([
        focal_length * x / z,
        focal_length * y / z
    ])
    """Why divide by z?
        This creates the perspective effect."""
    

def orthographic_projection(P):     #this boi is for depth
    x, y, z = P

    return np.array([x, y])


def camera_projection(P, C, R, focal_length):
    P_cam = world_to_camera(P, C, R)

    P_2D = perspective_projection(
        P_cam,
        focal_length
    )

    return P_2D



# ============================================================
# making function calling work ez boiiii part ^_~
# ============================================================


# this thing is just to make function calling ez
def project_vertices(vertices, camera_position, camera_target, focal_length):
    
    # Create the camera rotation matrix
    R = look_at_rotation(
        camera_position,
        camera_target
    )

    projected_vertices = []

    # Project each 3D vertex into 2D
    for vertex in vertices:

        # World space → camera space
        camera_point = world_to_camera(
            vertex,
            camera_position,
            R
        )

        # Camera space → 2D
        projected_point = perspective_projection(
            camera_point,
            focal_length
        )

        projected_vertices.append(projected_point)

    return np.array(projected_vertices)



# ============================================================
# this is to determine direction of camera part ^_~
# ============================================================


def look_at_rotation(camera_position, target):
    direction = target - camera_position
    direction = direction / np.linalg.norm(direction)

    forward = direction

    up = np.array([0.0, 1.0, 0.0])

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    """
    right       → X-axis
    up          → Y-axis
    forward     → Z-axis
    """

    # Store the camera axes as columns of the rotation matrix
    R = np.column_stack([
        right,
        up,
        forward
    ])

    return R



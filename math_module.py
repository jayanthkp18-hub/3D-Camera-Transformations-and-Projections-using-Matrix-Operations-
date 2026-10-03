import numpy as np

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





def homogeneous_point(x, y, z):
    
    return np.array([x, y, z, 1])


def P_camera(P,C,R):
    relative = P - C
    
    return R.T @ relative           #this thingy is P_cam=Rt*(P-C)


def is_orthogonal(R):
    I = np.eye(3)
    
    # checking for this boi --> Rt.R = I
    # allclose checks if all elements of the two arrays are equal within a tolerance
    return np.allclose(R.T @ R, I)
    
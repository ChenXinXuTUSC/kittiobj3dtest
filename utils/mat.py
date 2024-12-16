import numpy as np

def R_mat(angle: float, axis, radian: bool=True):
    """
    generate the 3x3 rotation matrix with the specified axis and angle

    - param angle: rotation angle, in radian
    - param axis: rotation axis, [1,3] or [3]
    - return: rotation matrix R [3,3]
    """
    # angle to radian
    angle_rad = angle
    if not radian:
        angle_rad = np.radians(angle)

    # normalize rotation axis
    if type(axis) is not np.ndarray:
        axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # compute each component
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1 - c

    # build the rotation matrix
    rotation_matrix = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])

    return rotation_matrix


def T_vec(values: list):
    return np.array(values).reshape((-1, 1))

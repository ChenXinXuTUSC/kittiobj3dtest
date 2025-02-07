import numpy as np

import utils

def snapshot_spherical(
    _points: np.ndarray,
    _labels: np.ndarray,
    center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    uppvec: np.ndarray = np.array([0.0, 0.0, 1.0]),
    toward: np.ndarray = np.array([1.0, 0.0, 0.0]),
    fov_v: float = None, # vertical
    fov_h: float = None, # default to 90 degree
    img_h: int = 64, # vertical / height
    img_w: int = 512 # horizontal / width
):
    '''
    Spherical Projection with specified range of view and center

    Params
    -
        - points: np.ndarray, [N, 3] coordinates
        - labels: np.ndarray, [N, 1] point class type int
        - center: np.ndarray, [1, 3] center coord
        - uppvec: np.ndarray, [1, 3] world up vector
        - toward: np.ndarray, [1, 3] camera towards
        - fov_v: float, range of vertical field in degree
        - fov_h: float, range of horizontal field in degree
        - img_h: int, number of pixels on height
        - img_w: int, number of pixels on width

    Return
    -
        - fmap: np.ndarray, [5, C, H, W] shape [x, y, z, i, r] feature map
        - gdth: np.ndarray, [H, W] shape class label map of each pixel
        - rmap: list[list], back projection from pixel to pointcloud index
    '''

    points = np.copy(_points)
    labels = np.copy(_labels)
    coords = points[:, :3]

    assert len(center) == 3, "not in 3d?"
    assert len(uppvec) == 3, "not in 3d?"
    assert len(toward) == 3, "not in 3d?"

    if not isinstance(center, np.ndarray):
        center = np.array(center)
    if not isinstance(uppvec, np.ndarray):
        uppvec = np.array(uppvec)
    if not isinstance(toward, np.ndarray):
        toward = np.array(toward)

    uppvec = uppvec / np.linalg.norm(uppvec)
    toward = toward / np.linalg.norm(toward)
    
    # transform point cloud
    coords -= center

    # align uppvec to z-axis
    R_u2z = utils.A2B_R(uppvec, [0, 0, 1]) # treat z-axis as up axis
    coords = np.matmul(R_u2z, coords.T).T
    # align towrad to x-axis
    toward = np.dot(R_u2z, toward)
    toward[2] = 0.0 # projected to x-y plane
    R_t2x = utils.A2B_R(toward, [1, 0, 0])
    coords = np.matmul(R_t2x, coords.T).T

    radian_xy = np.arctan2(coords[:, 1], coords[:, 0]) # tan(y/x)
    radian_zz = np.arctan2(coords[:, 2], np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)) # tan(r/z)
    
    if fov_v is None:
        fov_v = radian_zz.max() - radian_zz.min()
    if fov_h is None:
        fov_h = radian_xy.max() - radian_xy.min()
    
    delta_h = fov_v / img_h # vertical // height
    delta_w = fov_h / img_w # horizontal // width
    

    img_coord_h = (radian_zz / delta_h).astype(np.int32)
    img_coord_w = (radian_xy / delta_w).astype(np.int32)
    
    mask = np.full((len(coords),), True, dtype=bool)
    # mask = mask & (radian_zz > -fov_v / 2) & (radian_zz < +fov_v / 2)
    mask = mask & (radian_xy > -fov_h / 2) & (radian_xy < +fov_h / 2)
    
    # # 创建一个Open3D的点云对象
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(coords[mask])
    # o3d.io.write_point_cloud("dbug.ply", point_cloud)
    
    img_coord_h -= img_coord_h[mask].min()
    img_coord_w -= img_coord_w[mask].min()

    used = np.array(list(range(len(coords))))[mask]
    # print("retained ratio", mask.astype(np.int32).sum() / len(mask))

    gdth = np.zeros((img_h, img_w), dtype=np.int32)
    fmap = np.zeros((5, img_h, img_w))
    # rmap is for back projection usage
    rmap = [[[] for _ in range(img_w)] for _ in range(img_h)]

    # feature vector [x, y, z, i, r], shape [C, H, W]
    for i in used:
        # `i` is already the original index of points
        rmap[img_coord_h[i]][img_coord_w[i]].append(i)

        # select the point with nearest range as the feature pixel
        dist = np.linalg.norm(coords[i])
        feat = np.array([*points[i], dist])
        # empty pixel or a nearer pixel lead to an update
        if fmap[4, img_coord_h[i], img_coord_w[i]] == 0 or dist < fmap[4, img_coord_h[i], img_coord_w[i]]:
            fmap[:, img_coord_h[i], img_coord_w[i]] = feat
            gdth[img_coord_h[i], img_coord_w[i]] = labels[i]
    
    return fmap, gdth, rmap

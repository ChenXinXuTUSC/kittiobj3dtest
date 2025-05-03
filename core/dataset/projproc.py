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
    rmap = np.stack([img_coord_h, img_coord_w], axis=0)
    # feature vector [x, y, z, i, r], shape [C, H, W]
    for i in used:
        # select the point with nearest range as the feature pixel
        dist = np.linalg.norm(coords[i])
        feat = np.array([*points[i], dist])
        # empty pixel or a samller depth value
        # lead to an update pixel feature
        if fmap[4, img_coord_h[i], img_coord_w[i]] == 0 or dist < fmap[4, img_coord_h[i], img_coord_w[i]]:
            fmap[:, img_coord_h[i], img_coord_w[i]] = feat
            gdth[img_coord_h[i], img_coord_w[i]] = labels[i]
    
    return fmap, gdth, rmap

def snapshot_orthognal(
        points: np.ndarray, labels: np.ndarray,
        img_h=64, img_w=64,
        proj_axis=np.array([0, 0, 1]), # 投影方向轴
        proj_center=np.array([0, 0, 0]), # 投影中心点
        x_range=None, y_range=None
    ):
    """正交投影点云到像素图像
    
    Args:
        points: 点云坐标, shape [N, 3]
        labels: 点云类别标签, shape [N]
        img_h: 投影图像高度
        img_w: 投影图像宽度
        proj_axis: 投影方向轴向量, 会被归一化
        proj_center: 投影中心点坐标
        x_range: 投影平面x轴投影范围, (min_x, max_x)
        y_range: 投影平面y轴投影范围, (min_y, max_y)
    
    Returns:
        gdth: 投影图像标签, shape [H, W]
        rmap: 每个像素对应的原始点云索引列表
    """
    coords = points.copy()
    
    # 将点云平移到投影中心
    coords = coords - proj_center
    
    # 归一化投影轴方向
    proj_axis = proj_axis / np.linalg.norm(proj_axis)
    
    # 计算投影平面的基向量
    # 选择任意与投影轴不平行的向量作为辅助向量
    aux_vec = np.array([1, 0, 0]) if not np.allclose(proj_axis, [1, 0, 0]) else np.array([0, 1, 0])
    # 计算投影平面的x轴方向(右方向)
    basis_x = np.cross(proj_axis, aux_vec)
    basis_x = basis_x / np.linalg.norm(basis_x)
    # 计算投影平面的y轴方向(上方向)
    basis_y = np.cross(proj_axis, basis_x)
    basis_y = basis_y / np.linalg.norm(basis_y)
    
    # 将点云投影到新的坐标系
    proj_x = np.dot(coords, basis_x)
    proj_y = np.dot(coords, basis_y)
    proj_z = np.dot(coords, proj_axis)
    
    if x_range is None:
        x_range = (proj_x.min(), proj_x.max())
    if y_range is None:
        y_range = (proj_y.min(), proj_y.max())
        
    delta_x = (x_range[1] - x_range[0]) / img_w
    delta_y = (y_range[1] - y_range[0]) / img_h
    
    img_coord_h = ((proj_y - y_range[0]) / delta_y).astype(np.int32)
    img_coord_w = ((proj_x - x_range[0]) / delta_x).astype(np.int32)
    
    mask = np.full((len(coords),), True, dtype=bool)
    mask = mask & (img_coord_h >= 0) & (img_coord_h < img_h)
    mask = mask & (img_coord_w >= 0) & (img_coord_w < img_w)
    
    used = np.array(list(range(len(coords))))[mask]
    
    gdth = np.zeros((img_h, img_w), dtype=np.int32)
    rmap = [[[] for _ in range(img_w)] for _ in range(img_h)]
    
    for i in used:
        rmap[img_coord_h[i]][img_coord_w[i]].append(i)
        
        # 选择投影轴方向上距离最近的点作为该像素的标签
        z_val = proj_z[i]
        curr_z = gdth[img_coord_h[i], img_coord_w[i]]
        if curr_z == 0 or z_val < curr_z:
            gdth[img_coord_h[i], img_coord_w[i]] = labels[i]
            
    return gdth, rmap

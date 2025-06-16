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

	gdth = np.zeros((img_h, img_w), dtype=np.int64)
	fmap = np.zeros((5, img_h, img_w))
	rmap = np.column_stack([points, img_coord_h, img_coord_w])
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

def snapshot_voxel(
	coords: np.ndarray, labels: np.ndarray,
	voxel_size: float, resolution: list, 
	cam_pos, eye_pos
):
	'''
	Params
	-
	* coords: np.ndarray, (N, 3), not point features, but point coordinates
	* labels: np.ndarray, (N, )
	* voxel_size: float
	* resolution: list, tuple - (H, W)
	* cam_pos: np.ndarray, list, tuple - initial camera position
	* eye_pos: np.ndarray, list, tuple	- where camera towards
	Return
	-
	* feat_map: np.ndarray, (H, W) - feature map with each pixel labeled class index
	'''
	if not isinstance(coords, np.ndarray):
		raise TypeError("coords must be np.ndarray")
	if not isinstance(labels, np.ndarray):
		raise TypeError("labels must be np.ndarray")
	if not isinstance(voxel_size, float):
		raise TypeError("voxel_size must be float")
	if not isinstance(resolution, np.ndarray):
		if isinstance(resolution, list) or isinstance(resolution, tuple):
			resolution = np.array(resolution, dtype=np.int32)
		else:
			raise TypeError("cam_pos must be np.ndarray/list/tuple")
	if not isinstance(cam_pos, np.ndarray):
		if isinstance(cam_pos, list) or isinstance(cam_pos, tuple):
			cam_pos = np.array(cam_pos, dtype=np.float32)
		else:
			raise TypeError("cam_pos must be np.ndarray/list/tuple")
	if not isinstance(eye_pos, np.ndarray):
		if isinstance(eye_pos, list) or isinstance(eye_pos, tuple):
			eye_pos = np.array(eye_pos, dtype=np.float32)
		else:
			raise TypeError("eye_pos must be np.ndarray/list/tuple")

	import utils
	# make a copy of the original points
	coords = coords.copy()
	orgidx = np.arange(len(coords))

	# 移动视点至投影中心
	coords  -= eye_pos
	cam_pos -= eye_pos
	cam_pos /= np.linalg.norm(cam_pos)


	# cam_pos 现在就是投影轴，设 cam_pos 为 z 轴，投影平面为 x-y 平面，将 cam_pos 变换至 [0,0,1] z 轴
	R = utils.A2B_R(cam_pos, np.array([0,0,1]))
	# 与 X-Y 平面轴对齐
	coords = coords @ R.T
	
	# 根据 range 限定的范围，裁剪点云（给定以 cam_pos 为中心的长宽，单位为米，只裁剪 X-Y 平面，Z 轴无限制）
	coords_vxlzed = (coords // voxel_size).astype(np.int32)
	# 保留拍摄栅格范围内的体素
	mask  = np.abs(coords_vxlzed[:, 0]) < (resolution[0] // 2)
	mask &= np.abs(coords_vxlzed[:, 1]) < (resolution[1] // 2)

	coords_vxlzed_clipped = coords_vxlzed[mask]
	labels_clipped = labels[mask]
	orgidx_clipped = orgidx[mask]

	# 转换为屏幕栅格坐标系
	corner = coords_vxlzed_clipped.min(axis=0)
	coords_vxlzed_clipped = coords_vxlzed_clipped - corner
	
	# voxels = np.zeros((*(coords_vxlzed.max(axis=0) + 1), labels.max() + 1), dtype=np.int32)
	voxels = np.zeros((
		resolution[0],
		resolution[1],
		coords_vxlzed_clipped.max(axis=0)[2] + 1,
		labels.max() + 1
	), dtype=np.int32)
	np.add.at(voxels, (*coords_vxlzed_clipped.T, labels_clipped), 1)

	# 每个体素的类别取类别计数最高的那个类别
	voxels = voxels.argmax(axis=-1)

	# 1. 提取非空体素坐标 nz_idxs = non_zero_indices
	nz_idxs = np.nonzero(voxels)
	# 以 Z 轴投影方向平面上体素横纵坐标
	nz_idxs_xy_tuple = np.array(
		list(zip(nz_idxs[0], nz_idxs[1])), 
		dtype=[('x', 'int'), ('y', 'int')]
	)
	nz_idxs_xyz_tuple = np.array(
		list(zip(nz_idxs[0], nz_idxs[1], nz_idxs[2])),
		dtype=[('x', 'int'), ('y', 'int'), ('z', 'int')]
	)
	# 2. 提取非空体素坐标中的第一个非重复坐标
	_, nz_nz_idxs = np.unique(nz_idxs_xy_tuple, return_index=True)

	data = np.zeros((4, *(voxels.shape[:2])), dtype=np.int32) # 特征包括 (x,y,z,d) 也即原始三维空间位置和投影深度
	gdth = np.zeros(voxels.shape[:2], dtype=np.int32)

	# 原始空间位置
	origin_voxel_coords = np.stack(nz_idxs).T + corner
	origin_voxel_coords = origin_voxel_coords @ R

	data[0:3, nz_idxs[0][nz_nz_idxs], nz_idxs[1][nz_nz_idxs]] = origin_voxel_coords[nz_nz_idxs].T
	# data[0][nz_idxs[0][nz_nz_idxs], nz_idxs[1][nz_nz_idxs]] = nz_idxs[0][nz_nz_idxs] - corner[0]
	# data[1][nz_idxs[0][nz_nz_idxs], nz_idxs[1][nz_nz_idxs]] = nz_idxs[1][nz_nz_idxs] - corner[1]
	# data[2][nz_idxs[0][nz_nz_idxs], nz_idxs[1][nz_nz_idxs]] = nz_idxs[2][nz_nz_idxs] - corner[2]
	# 构造深度图，Z 轴索引等价于深度
	data[3][nz_idxs[0][nz_nz_idxs], nz_idxs[1][nz_nz_idxs]] = nz_idxs[2][nz_nz_idxs]

	# 构造语义分割图
	gdth[
		nz_idxs[0][nz_nz_idxs],
		nz_idxs[1][nz_nz_idxs]
	] = voxels [
		nz_idxs[0][nz_nz_idxs],
		nz_idxs[1][nz_nz_idxs],
		nz_idxs[2][nz_nz_idxs]
	]

	# 构造原始点云索引索引到投影图像素的映射
	coords_vxlzed_clipped_xyz_tuple = np.array(
		list(zip(coords_vxlzed_clipped[:, 0], coords_vxlzed_clipped[:, 1], coords_vxlzed_clipped[:, 2])),
		dtype=[('x', 'int'), ('y', 'int'), ('z', 'int')]
	)
	# 掩码标记被摄像头拍摄到的一面的点云所属的表面体素
	mask = np.isin(coords_vxlzed_clipped_xyz_tuple, nz_idxs_xyz_tuple[nz_nz_idxs])
	rmap = np.hstack([orgidx_clipped.reshape(-1, 1)[mask], coords_vxlzed_clipped[mask, :2]])

	return data, gdth, rmap

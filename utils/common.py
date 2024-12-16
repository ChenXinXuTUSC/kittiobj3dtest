import os
import os.path as osp

import struct
import numpy as np
import open3d as o3d

def read_kitti_lidar_bin(file_path):
    size = os.path.getsize(file_path)
    point_num = int(size / 16)
    assert point_num * 16 == size, "invalid binary structure"

    lidar_pt_list = []
    with open(file_path, "rb") as f:
        bin_data = None
        while True:
            bin_data = f.read(4)
            if len(bin_data) < 4:
                break
            lidar_pt_list.append(struct.unpack('f', bin_data))
    return np.array(lidar_pt_list).reshape((-1, 4))

def save_pcd(points: np.ndarray, colors: np.ndarray=None, ds_size: float=0.05, out_name: str="output"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_ds = pcd.voxel_down_sample(ds_size)
    o3d.io.write_point_cloud(f"{out_name}.ply", pcd_ds)

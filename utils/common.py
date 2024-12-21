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


def fill_blank(img: np.ndarray, trh: float, num_valid: int):
    IMG_H, IMG_W = img.shape

    img_filled = np.copy(img)

    dirs = [
        [ 0,  1],
        [ 0, -1],
        [ 1,  0],
        [-1,  0],
        [ 1,  1],
        [ 1, -1],
        [-1,  1],
        [-1, -1]
    ]

    def in_bound(i: int, j: int) -> bool:
        if i < 0 or i >= IMG_H:
            return False
        if j < 0 or j >= IMG_W:
            return False
        return True

    for i in range(IMG_H):
        for j in range(IMG_W):
            if img[i][j] >= trh:
                continue
            pix = []
            for d in dirs:
                u = i + d[0]
                v = j + d[1]
                if in_bound(u, v) and img[u][v] > trh:
                    pix.append(img[u][v])
            if len(pix) >= num_valid:
                img_filled[i][j] = sum(pix) / len(pix)
    
    return img_filled

def normalized_fmap(fmap: np.ndarray, cidx: list):
    """
    对 [C, H, W] 的特征图的指定通道进行归一化。
    
    - param feature_map: 形状为 [C, H, W] 的 NumPy ndarray。
    - return: 归一化后的特征图，形状为 [C, H, W]。
    """
    
    # 对每个通道进行归一化
    normalized_feature_map = np.zeros_like(fmap, dtype=np.float32)
    for c in cidx:
        # 获取当前通道的特征图
        channel = fmap[c]
        
        # 计算当前通道的最小值和最大值
        min_value = np.min(channel)
        max_value = np.max(channel)
        
        # 归一化当前通道
        if max_value != min_value:  # 避免除以零
            normalized_channel = (channel - min_value) / (max_value - min_value)
        else:
            normalized_channel = np.zeros_like(channel, dtype=np.float32)
        
        # 将归一化后的通道赋值回特征图
        normalized_feature_map[c] = normalized_channel
    
    return normalized_feature_map


pallete = {
    "DontCare": [0, 0, 0],
    "Car": [255, 0, 0],
    "Cyclist": [0, 255, 0],
    "Pedestrian": [0, 0, 255],
    "Misc": [255, 255, 0],
    "Person_sitting": [255, 0, 255],
    "Tram": [0, 255, 255],
    "Truck": [255, 128, 0],
    "Van": [0, 128, 255]
}
def visualize_fmap(fmap: np.ndarray):
    '''
    - fmap: [H, W] shape
    '''
    img = np.zeros((*fmap.shape[:2], 3), dtype=np.uint8)
    for idx, (cls_name, color) in enumerate(pallete.items()):
        img[fmap == idx] = np.array(color)
    return img

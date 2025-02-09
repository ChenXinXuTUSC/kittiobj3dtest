import os
import os.path as osp

import time
import functools

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

def image_fill(img: np.ndarray, target: float, err: float, num_valid: int, mode: str="avg"):
    '''
    Fill the blank pixel with surrounding pixels' value

    Params
    -
    - img: np.ndarray, [H, W] shape tensor
    - target: which target value treated as blank
    - num_valid: number of not blank surrounding pixels
    - mode: str, [avg|cnt] average values or pick the value with max exist counts
    '''
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
            if abs(img[i][j] - target) >= err:
                continue
            pix = []
            for d in dirs:
                u = i + d[0]
                v = j + d[1]
                if in_bound(u, v) and abs(target - img[u][v]) > err:
                    pix.append(img[u][v])
            if len(pix) >= num_valid:
                if mode == "avg":
                    img_filled[i][j] = sum(pix) / len(pix)
                if mode == "cnt":
                    unique, count = np.unique(np.array(pix), return_counts=True)
                    img_filled[i][j] = unique[np.argmax(count)]
    
    return img_filled

def image_fill2(
    image: np.ndarray,
    empty: float,
    offst: float,
    num_valid: int,
):
    empty_coords = np.stack(np.where((image >= empty - offst) & (image <= empty + offst)), axis=-1)
    valid_coords = np.stack(np.where((image <  empty - offst) | (image >  empty + offst)), axis=-1)
    valid_coords_view = valid_coords.view([('', valid_coords.dtype)] * valid_coords.shape[1])

    offset_list = [
        [ 0, -1],
        [-1, -1],
        [-1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
        [ 1,  0],
        [ 1, -1],
    ]
    
    adj_sum = np.zeros((len(empty_coords, )))
    adj_cnt = np.zeros((len(empty_coords, )))

    for offset in offset_list:
        coords = empty_coords + offset
        coords_view = coords.view([('', coords.dtype)] * coords.shape[1])
        mask = np.in1d(coords_view, valid_coords_view)
        
        adj_sum[mask] += image[coords[mask][:, 0], coords[mask][:, 1]]
        adj_cnt += mask.astype(np.int32)
    
    mask = adj_cnt >= num_valid
    image[empty_coords[mask][:, 0], empty_coords[mask][:, 1]] = adj_sum[mask] / adj_cnt[mask]

    return image


def normalized_fmap(fmap: np.ndarray, cidx: list):
    """
    对 [C, H, W] 的特征图的指定通道进行归一化。
    
    - param feature_map: 形状为 [C, H, W] 的 NumPy ndarray。
    - return: 归一化后的特征图，形状为 [C, H, W]。
    """
    
    # 对每个通道进行归一化
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
        fmap[c] = normalized_channel
    
    return fmap


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

# deepseek
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        run_time = end_time - start_time  # 计算运行时间
        print(f"Function '{func.__name__}' executed in {run_time:.4f} seconds.")
        return result  # 返回目标函数的返回值
    return wrapper

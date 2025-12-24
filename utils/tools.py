import numpy as np
import torch
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import cv2
# import pandas as pd
# from scipy.ndimage import center_of_mass
from PIL import Image
# def uniform_sampling(masks, N=5):
#     n_points = []
#     for mask in masks:
#         if not isinstance(mask, np.ndarray):
#             mask = mask.cpu().numpy().astype(np.uint8)

#         # if mask.max() <= 1:  # 确保 0 和 255
#         #     mask = (mask > 0).astype(np.uint8) * 255
#         # print("Mask unique values in uniform_sampling:", np.unique(mask))
#         #判断掩码中是否有目标区域（值为1）
#         # if np.sum(mask==1)==0:
#         #     # raise ValueError("Mask contains no valid (non-zero) regions.")  # 如果没有目标区域，抛出错误
#         #     print("Warning: Mask contains no valid (non-zero) regions, skipping.")
#         #     continue  # 跳过该掩码，避免后续的错误
#         if np.sum(mask==1)==0:
#             indices=np.argwhere(mask==0)
#         elif np.sum(mask==0)==0:
#             indices=np.argwhere(mask==1)
#         else:
#             # indices = np.argwhere(mask == 255) # [y, x]
#             indices = np.argwhere(mask == 1)  # 寻找1或255

#          # **新增判断，防止空数组**
#         if len(indices) == 0:
#             print("Warning: No valid points found in mask, skipping.")
#             continue

#         # num_points = indices.shape[0]  # 获取目标区域的点的数量
#         # print(f"目标区域点的数量: {num_points}")

#         sampled_indices = np.random.choice(len(indices), N, replace=True)
#         sampled_points = np.flip(indices[sampled_indices], axis=1)
#         n_points.append(sampled_points.tolist())

#     return n_points

def uniform_sampling(mask, N):
    """
    从mask中均匀采样N个点，确保返回形状为 (N, 2)。当N=1时，取白色区域中心点。
    """
    n_points = []
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy().astype(np.uint8)

    print("打印掩码中的唯一值：", np.unique(mask))  # 打印掩码中的唯一值
    if np.sum(mask == 255) == 0:
        indices = np.argwhere(mask == 0)  # 采样背景点
    elif np.sum(mask == 0) == 0:
        indices = np.argwhere(mask == 255)  # 采样前景点
    else:
        indices = np.argwhere(mask == 255)  # 采样前景点

    # **防止空数组**
    if len(indices) == 0:
        print("Warning: No valid points found in mask, skipping.")
        n_points.append(np.zeros((N, 2)))  # 避免后续代码崩溃
    else:
        if N == 1:
            # 当N=1时，计算白色区域的中心点
            center_y = np.mean(indices[:, 0])  # y坐标均值
            center_x = np.mean(indices[:, 1])  # x坐标均值
            sampled_points = np.array([[center_x, center_y]])  # 形状为 (1, 2)
        else:
            # 其他情况下，随机均匀采样
            sampled_indices = np.random.choice(len(indices), N, replace=True)
            sampled_points = indices[sampled_indices]  # (N, 2), 顺序是 [y, x]
            sampled_points = np.flip(sampled_points, axis=1)  # 变为 [x, y]
        
        n_points.append(sampled_points)

    return np.array(n_points) # 返回 (N, 2) 或 (1, 2)

# import numpy as np

# def uniform_sampling(mask, N):
#     """
#     从mask中均匀采样N个点，确保返回形状为 (N, 2)
#     """
#     n_points = []
#     # for mask in masks:
#     if not isinstance(mask, np.ndarray):
#         mask = mask.cpu().numpy().astype(np.uint8)

#     print("打印掩码中的唯一值：", np.unique(mask))  # 打印掩码中的唯一值
#     if np.sum(mask == 255) == 0:
#         indices = np.argwhere(mask == 0)  # 采样背景点
#     elif np.sum(mask == 0) == 0:
#         indices = np.argwhere(mask == 255)  # 采样前景点
#     else:
#         indices = np.argwhere(mask == 255)  # 采样前景点

#     # **防止空数组**
#     if len(indices) == 0:
#         print("Warning: No valid points found in mask, skipping.")
#         n_points.append(np.zeros((N, 2)))  # 避免后续代码崩溃
#         # continue

#     # **确保采样数量固定**
#     sampled_indices = np.random.choice(len(indices), N, replace=True)
#     sampled_points = indices[sampled_indices]  # (N, 2), 但顺序是 [y, x]
#     sampled_points = np.flip(sampled_points, axis=1)  # 变为 [x, y]
#     n_points.append(sampled_points)

#     return np.array(n_points)

# def sample_black_gray_points(image, debug=False):
#     """
#     从三值图像（0=黑，128=灰，255=白）中采样黑色和灰色区域各一个点
#     返回形状为 (2, 2) 的数组，格式 [[x1,y1], [x2,y2]]
#     """
#     # 转换为numpy数组，并确保是单通道灰度图
#     if not isinstance(image, np.ndarray):
#         image = image.cpu().numpy().squeeze().astype(np.uint8)
    
#     # 确保输入是灰度图（处理意外情况）
#     if len(image.shape) == 3:
#         image = np.mean(image, axis=2).astype(np.uint8)
    
#     # 定义黑色（0）和灰色（128）区域掩码
#     black_mask = (image == 0)
#     gray_mask = (image == 128)
    
#     # 调试信息
#     if debug:
#         unique_values = np.unique(image)
#         print(f"[Debug] 图像唯一像素值: {unique_values}")
#         print(f"[Debug] 黑色像素数: {np.sum(black_mask)}, 灰色像素数: {np.sum(gray_mask)}")
    
#     # 采样黑色点（0）
#     black_coords = np.argwhere(black_mask)
#     if len(black_coords) == 0:
#         print("Warning: 未找到黑色像素，使用默认点 (0,0)")
#         black_point = np.array([[0, 0]])
#     else:
#         idx = np.random.choice(len(black_coords), 1)
#         black_point = black_coords[idx]  # 格式 [y, x]

#     # 采样灰色点（128）
#     gray_coords = np.argwhere(gray_mask)
#     if len(gray_coords) == 0:
#         print("Warning: 未找到灰色像素，使用默认点 (0,0)")
#         gray_point = np.array([[0, 0]])
#     else:
#         idx = np.random.choice(len(gray_coords), 1)
#         gray_point = gray_coords[idx]  # 格式 [y, x]

#     # 转换坐标顺序为 [x, y] 并合并
#     sampled_points = np.concatenate([
#         np.flip(black_point, axis=1),  # 黑色点 [y,x] → [x,y]
#         np.flip(gray_point, axis=1)    # 灰色点 [y,x] → [x,y]
#     ], axis=0)

#     return sampled_points

def sample_black_gray_pointsDISC(image, num_points, debug=False):
    """
    从三值图像（0=黑，128=灰，255=白）中采样黑色和灰色区域的点
    当num_points=1时，取黑色和灰色区域的中心点
    返回形状为 (num_points, 2) 的数组，格式 [[x1,y1], [x2,y2], ...]
    """
    # 转换为numpy数组，并确保是单通道灰度图
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy().squeeze().astype(np.uint8)
    
    # 确保输入是灰度图
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # 定义像素值
    black_value = 0
    gray_value = 128
    
    # 定义黑色和灰色区域掩码（合并）
    disc_mask = (image == black_value) | (image == gray_value)
    
    # 调试信息
    if debug:
        unique_values = np.unique(image)
        print(f"[Debug] 图像唯一像素值: {unique_values}")
        print(f"[Debug] 黑色像素值: {black_value}, 灰色像素值: {gray_value}")
        print(f"[Debug] 黑色+灰色像素数: {np.sum(disc_mask)}")
    
    # 采样点
    disc_coords = np.argwhere(disc_mask)
    if len(disc_coords) == 0:
        print("Warning: 未找到黑色或灰色像素，返回默认点")
        return np.zeros((num_points, 2), dtype=np.int32)
    
    # 当num_points=1时，取中心点
    if num_points == 1:
        center_y = np.mean(disc_coords[:, 0])  # y坐标均值
        center_x = np.mean(disc_coords[:, 1])  # x坐标均值
        sampled_points = np.array([[center_x, center_y]], dtype=np.int32)  # 格式 [x, y]
    else:
        # 随机选择指定数量的点
        num_available = len(disc_coords)
        indices = np.random.choice(num_available, min(num_points, num_available), replace=False)
        sampled_points = disc_coords[indices]  # 格式 [y, x]
        # 转换坐标顺序为 [x, y]
        sampled_points = sampled_points[:, ::-1].copy()  # 使用索引反转并复制
    
    # 如果采样点少于要求数量，填充默认点 (0,0)
    if len(sampled_points) < num_points:
        padding = np.zeros((num_points - len(sampled_points), 2), dtype=np.int32)
        sampled_points = np.concatenate([sampled_points, padding], axis=0)
    
    return sampled_points

def sample_black_gray_pointsCUP(image, num_points, debug=False):
    """
    从三值图像（0=黑，128=灰，255=白）中采样黑色区域的点
    当num_points=1时，取黑色区域的中心点
    返回形状为 (num_points, 2) 的数组，格式 [[x1,y1], [x2,y2], ...]
    """
    # 转换为numpy数组，并确保是单通道灰度图
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy().squeeze().astype(np.uint8)
    
    # 确保输入是灰度图
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # 定义像素值
    black_value = 0
    
    # 定义黑色区域掩码
    black_mask = (image == black_value)
    
    # 调试信息
    if debug:
        unique_values = np.unique(image)
        print(f"[Debug] 图像唯一像素值: {unique_values}")
        print(f"[Debug] 黑色像素值: {black_value}")
        print(f"[Debug] 黑色像素数: {np.sum(black_mask)}")
    
    # 采样点
    black_coords = np.argwhere(black_mask)
    if len(black_coords) == 0:
        print("Warning: 未找到黑色像素，返回默认点")
        return np.zeros((num_points, 2), dtype=np.int32)
    
    # 当num_points=1时，取中心点
    if num_points == 1:
        center_y = np.mean(black_coords[:, 0])  # y坐标均值
        center_x = np.mean(black_coords[:, 1])  # x坐标均值
        sampled_points = np.array([[center_x, center_y]], dtype=np.int32)  # 格式 [x, y]
    else:
        # 随机选择指定数量的点
        num_available = len(black_coords)
        indices = np.random.choice(num_available, min(num_points, num_available), replace=False)
        sampled_points = black_coords[indices]  # 格式 [y, x]
        # 转换坐标顺序为 [x, y]
        sampled_points = sampled_points[:, ::-1].copy()  # 使用索引反转并复制
    
    # 如果采样点少于要求数量，填充默认点 (0,0)
    if len(sampled_points) < num_points:
        padding = np.zeros((num_points - len(sampled_points), 2), dtype=np.int32)
        sampled_points = np.concatenate([sampled_points, padding], axis=0)
    
    return sampled_points

# def uniform_sampling(masks, N=5):
#     n_points = []
#     for mask in masks:
#         if not isinstance(mask, np.ndarray):
#             mask = mask.cpu().numpy()

#         # 判断掩码中是否有目标区域（值为1）
#         if np.sum(mask == 1) == 0 & np.sum(mask==255)==0:
#             # raise ValueError("Mask contains no valid (non-zero) regions.")  # 如果没有目标区域，抛出错误
#             print("Warning: Mask contains no valid (non-zero) regions, skipping.")
#             continue  # 跳过该掩码，避免后续的错误


#         # 获取掩码中目标区域（值为1）的索引
#         indices = np.argwhere(mask == 1)  # [y, x]

#         # 随机选择 N 个点
#         sampled_indices = np.random.choice(len(indices), N, replace=True)
#         sampled_points = np.flip(indices[sampled_indices], axis=1)  # 翻转索引，使其符合 [x, y] 的顺序
#         n_points.append(sampled_points.tolist())

#     return n_points


def get_multi_distance_points(input_point, mask, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[1], input_point[0]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_distance_point(new_points[:i + 1, :], mask)

    new_points = swap_xy(new_points)
    return new_points


def get_next_distance_point(input_points, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    input_points = np.array(input_points)

    indices = np.argwhere(mask == True)
    for x, y in indices:
        # print(x,y,input_points)
        distance = np.sum(np.sqrt((x - input_points[:, 0]) ** 2 + (y - input_points[:, 1]) ** 2))
        if max_distance < distance:
            max_distance_point = [x, y]
            max_distance = distance
    return max_distance_point


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points


# def k_means_sampling(mask, k):
#     points = np.argwhere(mask == 1) # [y, x]
#     points = np.flip(points, axis=1)

#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(points)
#     points = kmeans.cluster_centers_
#     return points


def get_point_prompt_max_dist(mask, num_points):
    n_points = []
    # for mask in masks:
    # mask_np = mask.cpu().numpy()
    mask_np=mask

    indices = np.argwhere(mask_np ==255 )
    random_index = np.random.choice(len(indices), 1)[0]

    first_point = [indices[random_index][1], indices[random_index][0]]
    new_points = get_multi_distance_points(first_point, mask_np, num_points - 1)
    n_points.append(new_points)

    return np.array(n_points)


def get_point_prompt_within_mask(mask, num_points, radius=50):
    n_points = []
    # for mask in masks:
    # 只取掩膜中值为255的白色区域
    mask_np = mask
    print("mask.shape in get_point_prompt_within_mask:", mask_np.shape)
    y, x = np.where(mask_np == 255)  # 获取所有白色区域的像素位置
    white_pixels = list(zip(y, x))  # 白色区域的像素坐标
    
    if len(white_pixels) == 0:
        n_points.append(np.zeros((num_points, 2)))  # 如果没有白色区域，则返回空点
        print("返回空点！")
        # continue

    # 计算白色区域的质心（中心点）
    center_y, center_x = np.mean(y), np.mean(x)

    points = []
    # 生成至少一个点在中心
    points.append([center_y, center_x])

    # 生成其他的点，且这些点都在白色区域内
    for _ in range(num_points - 1):
        # 随机选取一个点，确保在白色区域内
        point = white_pixels[np.random.randint(0, len(white_pixels))]
        points.append(point)

    n_points.append(np.array(points))  # 将生成的点添加到结果列表
    
    return np.array(n_points)
# #每个mask都是使用"均匀采样"来生成点。
# #从mask图像中生成point
# def get_point_prompts(gt_masks, num_points):
#     prompts = []
#     for mask in gt_masks:
#         # print(f"Mask size: {mask.size()}")
#         # print("mask unique values:", np.unique(mask.cpu().numpy()))
#         mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
#         # background_mask = (~mask.to(bool)).to(torch.uint8) * 255
#         po_points = uniform_sampling(mask, num_points)
#         na_points = uniform_sampling(torch.tensor(mask == 0, dtype=torch.float32), num_points)
#         # na_points = uniform_sampling((~mask.to(bool)).to(float), num_points)
#         # na_points = uniform_sampling((mask == 0).float(), num_points)
#         # na_points = uniform_sampling(background_mask, num_points)
#         print("po_points.shape:", np.array(po_points).shape)  # (18, 5, 2)?
#         print("na_points.shape:", np.array(na_points).shape)  # (224, 5, 2)?
#         print(f"po_points count: {len(po_points)}")
#         print(f"na_points count: {len(na_points)}")
#         po_point_coords = torch.tensor(po_points, device=mask.device)
#         na_point_coords = torch.tensor(na_points, device=mask.device)
#         print("po_point_coords shape:", po_point_coords.shape)
#         print("na_point_coords shape:", na_point_coords.shape)
#         print("前景区域总像素点数:", mask.sum().item())
#         print("背景区域总像素点数:", (~mask.to(bool)).sum().item())
#         print("mask unique values:", np.unique(mask.cpu().numpy()))
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), connectivity=8)
#         print("前景区域数:", num_labels - 1)  # 不包括背景
#         point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
#         print(f"point_coords shape: {point_coords.shape}")
#         po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=po_point_coords.device)
#         na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=na_point_coords.device)
#         point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
#         in_points = (point_coords, point_labels)
#         prompts.append(in_points)
#     return prompts

# def get_point_prompts(gt_masks, num_points):
#     """
#     生成前景/背景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
#     """
#     prompts = []
#     for mask in gt_masks:
#         mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
        
#         # **获取前景点**
#         po_points = uniform_sampling([mask.cpu().numpy()], num_points)
#         na_points = uniform_sampling([~mask.cpu().numpy()], num_points)  # 取反获取背景点

#         print("po_points.shape:", po_points.shape)  # (1, num_points, 2)
#         print("na_points.shape:", na_points.shape)  # (1, num_points, 2)

#         # **转换为 PyTorch tensor**
#         po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)
#         na_point_coords = torch.tensor(na_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

#         print("po_point_coords shape:", po_point_coords.shape)  # 预期 (num_points, 2)
#         print("na_point_coords shape:", na_point_coords.shape)  # 预期 (num_points, 2)

#         # **确保 point_coords 形状正确**
#         point_coords = torch.cat((po_point_coords, na_point_coords), dim=0)  # (2*num_points, 2)

#         # **创建对应的 labels**
#         po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device)
#         na_point_labels = torch.zeros(na_point_coords.shape[0], dtype=torch.int, device=na_point_coords.device)
#         point_labels = torch.cat((po_point_labels, na_point_labels), dim=0)  # (2*num_points,)

#         print(f"Final point_coords shape: {point_coords.shape}")  # 应该是 (2*num_points, 2)
#         print(f"Final point_labels shape: {point_labels.shape}")  # 应该是 (2*num_points,)

#         # **打包成 tuple**
#         in_points = (point_coords, point_labels)
#         prompts.append(in_points)

#     return prompts
def get_point_prompts_seg(gt_masks, num_points):
    """
    生成前景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
    """
    prompts = []
    for mask in gt_masks:
        # mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
        
        # **获取前景点**
        # po_points = get_point_prompt_within_mask([mask.cpu().numpy()], num_points)
        print("mask.shape in get_point_prompts:", mask.shape)
        po_points = uniform_sampling(mask, num_points)
        # po_points = sample_black_gray_points(mask, debug=True)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        # **转换为 PyTorch tensor**
        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

        print("po_point_coords shape:", po_point_coords.shape)  # 预期 (num_points, 2)

        # **创建对应的 labels**
        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device)  # 前景点标签为 1

        print(f"Final point_coords shape: {po_point_coords.shape}")  # 应该是 (num_points, 2)
        print(f"Final point_labels shape: {po_point_labels.shape}")  # 应该是 (num_points,)

        # **打包成 tuple**
        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)
        # point_color = (0, 255, 255)  # 采用红色标注点
        # radius = 5  # 设置点的大小
        # print("point_coords in use:", po_point_coords.shape)
        # mask=mask
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为三通道 RGB 图像
        # print("masks before circle:", mask.shape)
        # for (x, y) in po_point_coords:
        #     cv2.circle(mask, (int(x), int(y)), radius, point_color, -1)  # -1 表示填充圆
        # print("masks after circle:", mask.shape)
        # image_pil = Image.fromarray(mask)
        # image_pil.save('output_in_get_point_prompts.png')
        # exit()

    return prompts

def get_point_prompts_clasDISC(gt_masks, num_points):
    """
    生成前景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
    """
    prompts = []
    for mask in gt_masks:
        # mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
        
        # **获取前景点**
        # po_points = get_point_prompt_within_mask([mask.cpu().numpy()], num_points)
        print("mask.shape in get_point_prompts:", mask.shape)
        # po_points = uniform_sampling(mask, num_points)
        po_points = sample_black_gray_pointsDISC(mask, num_points, debug=True)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        # **转换为 PyTorch tensor**
        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

        print("po_point_coords shape:", po_point_coords.shape)  # 预期 (num_points, 2)

        # **创建对应的 labels**
        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device)  # 前景点标签为 1

        print(f"Final point_coords shape: {po_point_coords.shape}")  # 应该是 (num_points, 2)
        print(f"Final point_labels shape: {po_point_labels.shape}")  # 应该是 (num_points,)

        # **打包成 tuple**
        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)
        # point_color = (0, 255, 255)  # 采用红色标注点
        # radius = 5  # 设置点的大小
        # print("point_coords in use:", po_point_coords.shape)
        # mask=mask
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为三通道 RGB 图像
        # print("masks before circle:", mask.shape)
        # for (x, y) in po_point_coords:
        #     cv2.circle(mask, (int(x), int(y)), radius, point_color, -1)  # -1 表示填充圆
        # print("masks after circle:", mask.shape)
        # image_pil = Image.fromarray(mask)
        # image_pil.save('output_in_get_point_prompts.png')
        # exit()

    return prompts

def get_point_prompts_clasCUP(gt_masks, num_points):
    """
    生成前景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
    """
    prompts = []
    for mask in gt_masks:
        # mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
        
        # **获取前景点**
        # po_points = get_point_prompt_within_mask([mask.cpu().numpy()], num_points)
        print("mask.shape in get_point_prompts:", mask.shape)
        # po_points = uniform_sampling(mask, num_points)
        po_points = sample_black_gray_pointsCUP(mask, num_points, debug=True)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        # **转换为 PyTorch tensor**
        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

        print("po_point_coords shape:", po_point_coords.shape)  # 预期 (num_points, 2)

        # **创建对应的 labels**
        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device)  # 前景点标签为 1

        print(f"Final point_coords shape: {po_point_coords.shape}")  # 应该是 (num_points, 2)
        print(f"Final point_labels shape: {po_point_labels.shape}")  # 应该是 (num_points,)

        # **打包成 tuple**
        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)
        # point_color = (0, 255, 255)  # 采用红色标注点
        # radius = 5  # 设置点的大小
        # print("point_coords in use:", po_point_coords.shape)
        # mask=mask
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为三通道 RGB 图像
        # print("masks before circle:", mask.shape)
        # for (x, y) in po_point_coords:
        #     cv2.circle(mask, (int(x), int(y)), radius, point_color, -1)  # -1 表示填充圆
        # print("masks after circle:", mask.shape)
        # image_pil = Image.fromarray(mask)
        # image_pil.save('output_in_get_point_prompts.png')
        # exit()

    return prompts

def generate_bounding_boxes_from_mask(mask):
    """
    从二值掩码图像生成包围白色区域的边界框。
    :param mask: 输入的二值图像，白色区域为255，其他为0。
    :return: 包围白色区域的边界框列表，每个框格式为 [x_min, y_min, x_max, y_max]。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # 找到白色区域的坐标
    # 这里我们假设白色区域是255，黑色区域是0
    y_indices, x_indices = np.where(mask == 255)  # 获取白色区域的所有坐标

    if len(x_indices) == 0 or len(y_indices) == 0:
        return []  # 没有白色区域时返回空列表

    # 获取白色区域的最小和最大坐标，形成边界框
    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)

    bounding_box = [[x_min, y_min, x_max, y_max]]

    return bounding_box

def sample_black_gray_boxDISC(mask):
    """
    从三值掩码图像（0=黑，128=灰，255=白）生成包含黑色和灰色区域的最小边界框。
    :param mask: 输入的三值图像，黑色为0，灰色为128，白色为255。
    :return: 包含黑色和灰色区域的边界框列表，格式为 [[x_min, y_min, x_max, y_max]]，若无目标区域返回空列表。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # 定义黑色和灰色区域的掩码（合并）
    disc_mask = (mask == 0) | (mask == 128)
    
    # 找到黑色和灰色区域的坐标
    y_indices, x_indices = np.where(disc_mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return []  # 没有黑色或灰色区域时返回空列表
    
    # 获取最小和最大坐标，形成边界框
    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)
    
    bounding_box = [[x_min, y_min, x_max, y_max]]
    
    return bounding_box

def sample_black_gray_boxCUP(mask):
    """
    从三值掩码图像（0=黑，128=灰，255=白）生成包含黑色区域的最小边界框。
    :param mask: 输入的三值图像，黑色为0，灰色为128，白色为255。
    :return: 包含黑色区域的边界框列表，格式为 [[x_min, y_min, x_max, y_max]]，若无目标区域返回空列表。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # 定义黑色区域的掩码
    black_mask = (mask == 0)
    
    # 找到黑色区域的坐标
    y_indices, x_indices = np.where(black_mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return []  # 没有黑色区域时返回空列表
    
    # 获取最小和最大坐标，形成边界框
    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)
    
    bounding_box = [[x_min, y_min, x_max, y_max]]
    
    return bounding_box
        
def visualize_and_save_masks_refuge(masks, scores, output_dir, file_name_prefix):
    """
    可视化并保存掩码结果，将 (2, H, W) 的掩码保存为两张图片。
    
    参数:
        masks (numpy.ndarray): 预测的掩码数组，形状为 (2, H, W)。
        scores (list or numpy.ndarray): 每个掩码的分数。
        output_dir (str): 输出文件夹路径。
        file_name_prefix (str): 保存文件名的前缀。
    """
    # 检查输出目录是否存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保 masks 的形状为 (2, H, W)
    if masks.shape[0] != 2:
        print(f"Error: 期望 masks 形状为 (2, H, W)，实际为 {masks.shape}")
        return
    
    # 定义文件名
    file_names = [
        f"{file_name_prefix}_black.png",
        f"{file_name_prefix}_gray.png"
    ]
    
    # 对每个掩码及其分数进行保存
    for i, (mask, score, file_name) in enumerate(zip(masks, scores, file_names)):
        output_path = os.path.join(output_dir, file_name)
        
        # 检查文件是否已存在，若存在则覆盖
        if os.path.exists(output_path):
            print(f"文件 {file_name} 已存在，正在覆盖...")
        
        # 使用 matplotlib 保存图像
        plt.imsave(output_path, mask, cmap='gray')  # 使用灰度图保存掩码

def visualize_and_save_masks(masks, output_dir, img_name, expected_size=None):
    """
    保存单通道二值掩码为 PNG 图像，并确保不改变原图大小

    Args:
        masks (np.ndarray): 二值掩码，形状为 (1, H, W)
        output_dir (str): 掩码保存目录
        file_name_prefix (str): 输出文件名前缀
        expected_size (tuple): 可选，期望的图像大小 (W, H)，如与原图一致时可校验
    """
    import os
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    mask = masks[0]  # (H, W)

    if expected_size is not None:
        assert mask.shape[::-1] == expected_size, f"掩码尺寸 {mask.shape[::-1]} 与原图尺寸 {expected_size} 不一致"

    # 保存时不改变尺寸
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')  # 'L' 表示灰度图

    output_path = os.path.join(output_dir, f"{img_name}.png")
    mask_img.save(output_path)
    
import matplotlib.pyplot as plt

def save_cam_overlay(img_np, heatmap, cam_dir, file_prefix):
    """
    将热力图叠加到原图并保存

    参数:
        image_tensor: [3,H,W] Tensor, 范围 0~1
        heatmap: [H,W] Tensor, 热力图
        cam_dir: str, 保存目录
        file_prefix: str, 图片文件名前缀
    """
    # 确保目录存在
    os.makedirs(cam_dir, exist_ok=True)
    
    # 转成 numpy
    # img_np = image_tensor.permute(1,2,0).cpu().numpy()
    heatmap_np = heatmap.cpu().numpy()
    
    # 绘图
    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # 保存
    save_path = os.path.join(cam_dir, f"{file_prefix}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"CAM image saved to {save_path}")
    
import os
import csv
import torch
import copy
import numpy as np
# from torchsummary import summary

def freeze(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def momentum_update(student_model, teacher_model, momentum=0.99):
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        student_model.named_parameters(), teacher_model.named_parameters()
    ):
        if src_param.requires_grad:
            tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1 - momentum)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


def copy_model(model: torch.nn.Module):
    new_model = copy.deepcopy(model)
    freeze(new_model)
    return new_model


def create_csv(filename, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    if os.path.exists(filename):
        return 
    with open(filename, 'w') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head)
        csv_write.writeheader()


def write_csv(filename, csv_dict, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    with open(filename, 'a+') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head, extrasaction='ignore')
        csv_write.writerow(csv_dict)


def check_grad(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def check_equal(model1: torch.nn.Module, model2: torch.nn.Module):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            if not torch.allclose(param1, param2):
                print(f"{name1} is different")
            else:
                print(f"same")
        else:
            print("The models have different structures")


# def check_model(model):
#     return summary(model, (3, 1024, 1024), batch_size=1, device="cuda")


def reduce_instances(bboxes, gt_masks, max_nums=50):
    bboxes_ = []
    gt_masks_ = []
    for bbox, gt_mask in zip(bboxes, gt_masks):
        idx = np.arange(bbox.shape[0])
        np.random.shuffle(idx)
        bboxes_.append(bbox[idx[:max_nums]])
        gt_masks_.append(gt_mask[idx[:max_nums]])

    bboxes = bboxes_
    gt_masks = gt_masks_
    return bboxes, gt_masks

def compute_grad_cam(layer_name, output_dir_matrices, img_name, device):
    # 激活图和梯度的路径
    activation_path = os.path.join(output_dir_matrices, img_name, layer_name, "activation.npy")
    gradient_path = os.path.join(output_dir_matrices, img_name, layer_name, "gradient.npy")

    activation = np.load(activation_path)  # 从文件加载激活特征
    gradient = np.load(gradient_path)      # 从文件加载梯度
    print("Activation shape:", activation.shape)
    print("Gradient shape:", gradient.shape)

    # 将加载的 numpy 数组转换为 tensor
    activation = torch.tensor(activation).to(device)
    gradient = torch.tensor(gradient).to(device)

    # 计算 Grad-CAM 权重（全局平均池化）
    weights = torch.mean(gradient, dim=[1, 2], keepdim=True)
    print("weights.shape in compute_grad_cam:", weights.shape)
    
    # 计算 Grad-CAM 结果
    grad_cam_map = torch.sum(weights * activation, dim=-1)  # 线性组合
    print("grad_cam_map in compute_grad_cam at first:",grad_cam_map.shape)
    grad_cam_map = torch.relu(grad_cam_map)  # ReLU 确保非负
    
    # 确保维度合适
    grad_cam_map = grad_cam_map.squeeze(0)  # 删除批量维度，变为 (64, 64)
    
    # 归一化
    grad_cam_map = grad_cam_map - grad_cam_map.min()
    grad_cam_map = grad_cam_map / (grad_cam_map.max() + 1e-6)

    # # 如果你只想要一个通道，可以使用平均
    # print("grad_cam_shape in compute_grad_cam before mean:",grad_cam_map.shape)
    # grad_cam_map = grad_cam_map.mean(dim=1)  # 变为 (64, 768)
    # grad_cam_map = torch.relu(grad_cam_map)  # ReLU 确保非负
    # grad_cam_map = torch.max(grad_cam_map, dim=1)[0]  # 取每个像素的最大响应
    print("grad_cam_shape in compute_grad_cam:",grad_cam_map.shape)
    return grad_cam_map

def process_grad_cam_overlay(grad_cam_map, image, colormap=cv2.COLORMAP_JET, alpha=0.3, threshold=0.5):
    """处理 Grad-CAM map，使其通过 jet 可视化并叠加到原图上"""
    
    if np.min(grad_cam_map) < 0:
        print(f"Grad-CAM map 中包含负数，最小值：{np.min(grad_cam_map)}")

    # 转为 float32 并归一化到 [0,1]
    grad_cam_map = grad_cam_map.astype(np.float32)
    grad_cam_map = cv2.normalize(grad_cam_map, None, alpha=0, beta=1, 
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # gamma 调整增强对比度
    grad_cam_map = np.power(grad_cam_map, 0.5)
    grad_cam_map = cv2.normalize(grad_cam_map, None, 0, 1, cv2.NORM_MINMAX)

    # ✅ 阈值处理：抑制背景
    grad_cam_map[grad_cam_map < threshold] = 0

    # 调整大小
    h_img, w_img, _ = image.shape
    size_cam = grad_cam_map.shape[0]  # assume square

    if h_img >= w_img:
        scale_factor = h_img / size_cam
        new_w = int(size_cam * scale_factor)
        new_h = h_img
    else:
        scale_factor = w_img / size_cam
        new_h = int(size_cam * scale_factor)
        new_w = w_img

    grad_cam_resized = cv2.resize(grad_cam_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    grad_cam_cropped = grad_cam_resized[:h_img, :w_img]

    # 映射到 0~255 再转为热力图（Jet）
    heatmap = (grad_cam_cropped * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # 如果原图是灰度图，先转换为三通道
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 叠加热力图到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay

def overlay_img(image, pred_masks, masks, grad_cam_map, bounding_boxes, colormap=cv2.COLORMAP_HOT, box_color=(0, 255, 0), thickness=2):
    """
    在图像上叠加边界框并结合Grad-CAM热力图
    :param image: 输入图像
    :param bounding_boxes: 边界框列表，每个框格式为 [x_min, y_min, x_max, y_max]
    :param grad_cam_map: Grad-CAM热力图
    :param alpha: 透明度，默认为 0.5
    :param colormap: 热力图的颜色映射，默认为 cv2.COLORMAP_HOT
    :param threshold: 热力图阈值，用于抑制低值部分
    :param box_color: 边界框的颜色，默认为绿色
    :param thickness: 边界框线条的粗细，默认为 2
    :return: 叠加了边界框的图像
    """
    if isinstance(grad_cam_map, torch.Tensor):
        grad_cam_map = grad_cam_map.detach().cpu().numpy()
    # 确保 image 是 uint8 类型
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 如果 image 是单通道，转换为 3 通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 调整 pred_masks 大小
    pred_mask_uint8 = (pred_masks[0] * 255).astype(np.uint8)  # pred_masks 是 (1, 767, 1022)，取第 0 维
    pred_mask_resized = pred_mask_uint8
    pred_contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_uint8 = (masks[0] * 255).astype(np.uint8)  # masks 是 (1, h, w)，取第 0 维
    mask_resized = mask_uint8
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    heatmap = process_grad_cam_overlay(grad_cam_map, image)

    # # 在热力图上绘制轮廓
    # outline_color = (0, 255, 0)  # 绿色绘制标准框
    # thickness = 2
    # cv2.drawContours(heatmap, contours, -1, outline_color, thickness)
    # # 在热力图上绘制轮廓
    # outline_color = (255, 0, 0)  # 蓝色绘制预测框
    # thickness = 2
    # cv2.drawContours(heatmap, pred_contours, -1, outline_color, thickness)

    # # 处理每个边界框
    # for bbox in bounding_boxes:
    #     x_min, y_min, x_max, y_max = bbox
    #     cv2.rectangle(heatmap, (x_min, y_min), (x_max, y_max), box_color, thickness)

    return heatmap

def save_grad_cam_image(output_dir, img_name, layer_idx, overlayed_img):
    # 创建以 batch_idx 命名的文件夹
    batch_dir = os.path.join(output_dir, "heatmap", img_name)
    os.makedirs(batch_dir, exist_ok=True)

    # 构建文件路径并保存图像
    file_path = os.path.join(batch_dir, f"layer_{layer_idx}.png")
    cv2.imwrite(file_path, overlayed_img)
    

def compute_grad_cam_for_compare_gradient(layer_name, output_dir_matrices, img_name, device):
    # 激活图和梯度的路径
    activation_path = os.path.join(output_dir_matrices, img_name, layer_name, "activation.npy")
    gradient_path = os.path.join(output_dir_matrices, img_name, layer_name, "gradient.npy")

    activation = np.load(activation_path)  # 从文件加载激活特征
    gradient = np.load(gradient_path)      # 从文件加载梯度

    # 将加载的 numpy 数组转换为 tensor
    activation = torch.tensor(activation).to(device)
    gradient = torch.tensor(gradient).to(device)

    # 计算 Grad-CAM 权重（全局平均池化）
    weights = torch.mean(gradient, dim=[1, 2], keepdim=True)
    
    # 计算 Grad-CAM 结果
    grad_cam_map = torch.sum(weights * activation, dim=-1)  # 线性组合
    print("grad_cam_map in compute_grad_cam at first:",grad_cam_map.shape)
    grad_cam_map=grad_cam_map.cpu().numpy()

    # grad_cam_map = torch.relu(grad_cam_map)  # ReLU 确保非负
    
    # 确保维度合适
    grad_cam_map = grad_cam_map.squeeze(0)  # 删除批量维度，变为 (64, 768)
    
    # Z-score归一化
    grad_cam_map = (grad_cam_map - grad_cam_map.mean()) / (grad_cam_map.std() + 1e-6)

    # # 如果你只想要一个通道，可以使用平均
    # print("grad_cam_shape in compute_grad_cam before mean:",grad_cam_map.shape)
    # grad_cam_map = grad_cam_map.mean(dim=1)  # 变为 (64, 768)
    # grad_cam_map = torch.relu(grad_cam_map)  # ReLU 确保非负
    # grad_cam_map = torch.max(grad_cam_map, dim=1)[0]  # 取每个像素的最大响应
    print("grad_cam_shape in compute_grad_cam:",grad_cam_map.shape)

    return grad_cam_map

def process_grad_cam(grad_cam_map, image, colormap=cv2.COLORMAP_HOT):
    """处理 Grad-CAM map，使其可视化"""
    if np.min(grad_cam_map) < 0:
        print(f"grad-cam-map图中梯度包含负数，最小值：{np.min(grad_cam_map)}")

    # 处理 Grad-CAM
    grad_cam_map = grad_cam_map.astype(np.float32)
    grad_cam_map = cv2.normalize(grad_cam_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # 增强对比度
    grad_cam_map = np.power(grad_cam_map, 0.5)  # Gamma 变换
    grad_cam_map = cv2.normalize(grad_cam_map, None, 0, 1, cv2.NORM_MINMAX)

    # 插值法调整 grad_cam_map 尺寸
    h, w, c= image.shape
    target_height = int((h / w) * 64)
    grad_cam_map = grad_cam_map[:target_height, :]  # 裁剪掉多余的部分
    grad_cam_resized = cv2.resize(grad_cam_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # # 生成热力图
    heatmap = (grad_cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)   

    return heatmap

def extract_grad_from_box(grad_cam_heatmap, box_coords, image):
    """在 1022x767 的 Grad-CAM 热力图上计算框内外的梯度均值""" 
    H, W, C= image.shape # 处理后的热力图尺寸
    
    # 确保 box_coords 是二维数组，提取第一个框的坐标
    if isinstance(box_coords, (list, np.ndarray)) and len(box_coords) > 0:
        box_coords = box_coords[0]  # 取第一个框
    else:
        raise ValueError("box_coords 格式错误，应该是 [[x1, y1, x2, y2]] 的形式")

    # 解析框坐标
    x1, y1, x2, y2 = map(int, box_coords)

    # 确保坐标不越界
    x1, x2 = max(0, x1), min(W - 1, x2)
    y1, y2 = max(0, y1), min(H - 1, y2)

    # 提取框内的梯度值
    inside_grad = grad_cam_heatmap[y1:y2, x1:x2]
    inside_grad_mean = np.mean(inside_grad) if inside_grad.size > 0 else 0

    # 提取框外的梯度值
    outside_mask = np.ones_like(grad_cam_heatmap, dtype=bool)
    outside_mask[y1:y2, x1:x2] = False
    outside_grad_mean = np.mean(grad_cam_heatmap[outside_mask]) if outside_mask.any() else 0

    return inside_grad_mean, outside_grad_mean

def collect_gradients_box(output_dir, img_name, box_coords, device, image):
    """
    仅收集每一层框内外的梯度均值，不进行绘图。
    返回格式：List of (img_name, layer_label, inside_mean, outside_mean)
    """
    original_layers = [2, 5, 8, 11]
    layer_labels = [3, 6, 9, 12]
    output_dir_matrices = os.path.join(output_dir, "matrices")

    result_list = []

    for idx, label in zip(original_layers, layer_labels):
        grad_cam_map = compute_grad_cam_for_compare_gradient(f"layer_{idx}", output_dir_matrices, img_name, device)
        heatmap = process_grad_cam(grad_cam_map, image)
        inside_mean, outside_mean = extract_grad_from_box(heatmap, box_coords, image)

        max_value = max(inside_mean, outside_mean)
        inside_norm = inside_mean / max_value if max_value != 0 else 0
        outside_norm = outside_mean / max_value if max_value != 0 else 0

        result_list.append((img_name, label, inside_norm, outside_norm))

    return result_list


import os
import pandas as pd
import matplotlib.pyplot as plt

def visualize_all_gradients(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # 强制转换为数值，避免报错
    df["InsideGradient"] = pd.to_numeric(df["InsideGradient"], errors="coerce")
    df["OutsideGradient"] = pd.to_numeric(df["OutsideGradient"], errors="coerce")
    df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")

    # 丢掉非法值
    df = df.dropna(subset=["Layer", "InsideGradient", "OutsideGradient"])

    # 将 Layer 映射为 Stage
    layer_to_stage = {3: 1, 6: 2, 9: 3, 12: 4}
    df["Stage"] = df["Layer"].map(layer_to_stage)

    # 丢掉没有映射成功的 Layer（例如不是 3/6/9/12 的）
    df = df.dropna(subset=["Stage"])

    # 分组并计算平均值
    grouped = df.groupby("Stage")[["InsideGradient", "OutsideGradient"]].mean()

    stages = grouped.index.tolist()
    inside_means = grouped["InsideGradient"].tolist()
    outside_means = grouped["OutsideGradient"].tolist()

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(stages, inside_means, marker="o", label="Inside Box", color="blue")
    plt.plot(stages, outside_means, marker="s", label="Outside Box", color="red")
    plt.xlabel("Stage Index")
    plt.ylabel("Average Activation per Region")
    plt.title("Comparison Across ISIC dataset")
    plt.xticks(stages, labels=[f"Stage {int(s)}" for s in stages])
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_gradients_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved dataset-level gradient plot: {output_path}")
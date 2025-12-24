import os
import torch
import lightning as L
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts, get_point_prompts_seg
from utils.tools import write_csv
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PIL import Image, ImageDraw

def calculate_dice_torch(gt_mask: torch.Tensor, pred_mask: torch.Tensor, smooth=1e-8):
    """
    自定义Dice，输入都是float二值Tensor，shape (H, W) 或 (batch, H, W)
    """
    gt_mask = gt_mask.float()
    pred_mask = pred_mask.float()

    intersection = torch.sum(gt_mask * pred_mask, dim=(-2, -1))
    union = torch.sum(gt_mask, dim=(-2, -1)) + torch.sum(pred_mask, dim=(-2, -1))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_prob = torch.sigmoid(pred_mask)
    pred_bin = (pred_prob >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points) 
        print("point!")
        # prompts = get_point_prompts_seg(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts

import torchvision.transforms.functional as TF

def save_image_with_boxes(img_tensor, boxes, save_path="vis.png"):
    """
    img_tensor: (1, 3, H, W) tensor
    boxes: List of boxes, each box is [x1, y1, x2, y2]
    """
    # 去掉 batch 维度，并将图像从 tensor 转为 PIL.Image
    img_tensor = img_tensor.squeeze(0).cpu()
    img = TF.to_pil_image(img_tensor)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 绘制每一个 box
    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        box = [int(x) for x in box]
        draw.rectangle(box, outline="red", width=3)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def validate(fabric: L.Fabric, cfg: Box, model: Model, load_datasets, name: str, iters: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    dice_scores = AverageMeter()  # 新增Dice指标
    saved_flag = False  # 控制只保存一张掩码图像
    print(2)
    train_dataloader, val_dataloader, test_dataloader = load_datasets(cfg, model.model.image_encoder.img_size)
    print(3)

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            print("img.shape:", images.shape)
            # # 取第一个图像和box
            # img = images[0]  # [3, H, W]
            # box_list = bboxes[0]  # 是多个框

            # save_image_with_boxes(img.unsqueeze(0), box_list, save_path="debug/vis_box.png")
            
            num_images = images.size(0)
            
            # ✅ 把 images 搬到 GPU 上
            images = images.to(fabric.device)
            
            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)
            print("Predicted mask min/max:", pred_masks[0].min().item(), pred_masks[0].max().item())

            for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                pred_bin = (pred_mask >= 0.5).float()

                # 保存一张预测掩码图
                if not saved_flag:
                    print("pred_bin shape:", pred_bin.shape)
                    # pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8) * 255
                    pred_np = pred_bin[0].cpu().numpy().astype(np.uint8) * 255  # 取出最内层图像数据
                    print("pred_np shape:", pred_np.shape)
                    save_dir = os.path.join(cfg.out_dir, "pred_masks")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "val_pred_mask.png")
                    Image.fromarray(pred_np).save(save_path)
                    fabric.print(f"✅ 已保存验证集预测掩码图像: {save_path}")
                    saved_flag = True  # 只保存一次

                if gt_mask.max() > 1:
                    gt_bin = (gt_mask == 255).float()
                else:
                    gt_bin = gt_mask.float()

                # 自定义Dice
                gt_bin = gt_bin.to(fabric.device)
                batch_dice = calculate_dice_torch(gt_bin, pred_bin).mean().item()

                # IoU 和 F1
                pred_mask = pred_mask.to(fabric.device)
                gt_mask = gt_mask.to(fabric.device)
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")

                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                dice_scores.update(batch_dice, num_images)

            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: -- Mean Dice: [{dice_scores.avg:.4f}] Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean Dice: [{dice_scores.avg:.4f}]')
    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "Mean IoU": f"{ious.avg:.4f}",
        "Mean F1": f"{f1_scores.avg:.4f}",
        "iters": iters
    }

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)

    model.train()
    return ious.avg, f1_scores.avg, dice_scores.avg

# def test(fabric: L.Fabric, cfg: Box, model: Model, load_datasets, name: str, iters: int = 0):
#     model.eval()
#     ious = AverageMeter()
#     f1_scores = AverageMeter()
#     dice_scores = AverageMeter()  # 新增Dice指标
#     saved_flag = False  # 控制只保存一张掩码图像
#     train_dataloader, val_dataloader, test_dataloader = load_datasets(cfg, model.model.image_encoder.img_size)

#     with torch.no_grad():
#         for iter, data in enumerate(test_dataloader):
#             images, bboxes, gt_masks = data
#             print("img.shape:", images.shape)
#             # 取第一个图像和box
#             img = images[0]  # [3, H, W]
#             box_list = bboxes[0]  # 是多个框

#             save_image_with_boxes(img.unsqueeze(0), box_list, save_path="debug/vis_box.png")
            
#             num_images = images.size(0)
            
#             # ✅ 把 images 搬到 GPU 上
#             images = images.to(fabric.device)
            
#             prompts = get_prompts(cfg, bboxes, gt_masks)
#             print("prompts:", prompts)

#             _, pred_masks, _, _ = model(images, prompts)
#             print("Predicted mask min/max:", pred_masks[0].min().item(), pred_masks[0].max().item())

#             for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
#                 pred_bin = (pred_mask >= 0.5).float()

#                 # 保存一张预测掩码图
#                 if not saved_flag:
#                     print("pred_bin shape:", pred_bin.shape)
#                     # pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8) * 255
#                     pred_np = pred_bin[0].cpu().numpy().astype(np.uint8) * 255  # 取出最内层图像数据
#                     print("pred_np shape:", pred_np.shape)
#                     save_dir = os.path.join(cfg.out_dir, "pred_masks")
#                     os.makedirs(save_dir, exist_ok=True)
#                     save_path = os.path.join(save_dir, "val_pred_mask.png")
#                     Image.fromarray(pred_np).save(save_path)
#                     fabric.print(f"✅ 已保存验证集预测掩码图像: {save_path}")
#                     saved_flag = True  # 只保存一次

#                 if gt_mask.max() > 1:
#                     gt_bin = (gt_mask == 255).float()
#                 else:
#                     gt_bin = gt_mask.float()

#                 # 自定义Dice
#                 gt_bin = gt_bin.to(fabric.device)
#                 batch_dice = calculate_dice_torch(gt_bin, pred_bin).mean().item()

#                 # IoU 和 F1
#                 pred_mask = pred_mask.to(fabric.device)
#                 gt_mask = gt_mask.to(fabric.device)
#                 batch_stats = smp.metrics.get_stats(
#                     pred_mask,
#                     gt_mask.int(),
#                     mode='binary',
#                     threshold=0.5,
#                 )
#                 batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
#                 batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")

#                 ious.update(batch_iou, num_images)
#                 f1_scores.update(batch_f1, num_images)
#                 dice_scores.update(batch_dice, num_images)

#             fabric.print(
#                 f'test: [{iters}] - [{iter}/{len(test_dataloader)}]: -- Mean Dice: [{dice_scores.avg:.4f}] Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
#             )
#             torch.cuda.empty_cache()

#     fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean Dice: [{dice_scores.avg:.4f}]')
#     csv_dict = {
#         "Name": name,
#         "Prompt": cfg.prompt,
#         "Mean Dice": f"{dice_scores.avg:.4f}",
#         "Mean IoU": f"{ious.avg:.4f}",
#         "Mean F1": f"{f1_scores.avg:.4f}",
#         "iters": iters
#     }

#     if fabric.global_rank == 0:
#         write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)

#     model.train()
#     return ious.avg, f1_scores.avg, dice_scores.avg

import os
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import torch
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

def save_image_with_points(img_tensor, coords, labels, save_path):
    """
    在图像上绘制点并保存。

    参数:
        img_tensor: torch.Tensor, shape [1, 3, H, W] 或 [3, H, W]
        coords: numpy array or tensor, shape [N, 2]，每行是 (x, y)
        labels: numpy array or tensor, shape [N]，1为正点，0为负点
        save_path: 保存路径
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # [3, H, W]

    # 转PIL图像，确保在cpu上
    img = TF.to_pil_image(img_tensor.cpu())

    draw = ImageDraw.Draw(img)

    for i in range(len(coords)):
        x, y = coords[i]
        x, y = float(x), float(y)
        r = 5  # 点半径

        if labels[i] == 1:
            fill_color = "green"
        else:
            fill_color = "red"

        # 画实心圆点
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_color, outline=None)

    img.save(save_path)


def save_image_with_boxes(img_tensor, box_list, save_path):
    """
    将 box_list 中的框画在图像上并保存。
    img_tensor: shape [1, 3, H, W] or [3, H, W]
    box_list: tensor of shape [N, 4] (x1, y1, x2, y2)
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img = TF.to_pil_image(img_tensor.cpu())
    draw = ImageDraw.Draw(img)
    for box in box_list:
        x1, y1, x2, y2 = map(float, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    img.save(save_path)

def test(fabric, cfg, tented_model, load_datasets, name: str, iters: int = 0):
    tented_model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    dice_scores = AverageMeter()

    train_dataloader, val_dataloader, test_dataloader = load_datasets(cfg, tented_model.model.model.image_encoder.img_size)

    with torch.no_grad():
        for iter, data in enumerate(test_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            # 创建保存目录
            visual_dir = os.path.join(cfg.out_dir, "visual")
            pred_mask_dir = os.path.join(cfg.out_dir, "pred_masks")
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(pred_mask_dir, exist_ok=True)

            # 将图像搬到 GPU
            images = images.to(fabric.device)

            # 获取 prompts
            prompts = get_prompts(cfg, bboxes, gt_masks)
            
            if iter == 2:
                # 可视化 prompts（只画第一个样本）
                img = images[0].cpu()
                box_list = prompts[0].cpu().double()  # [N, 4]
                vis_save_path = os.path.join(visual_dir, f"visual_iter{iter}.png")
                save_image_with_boxes(img, box_list, vis_save_path)
                
                print("img.shape:", images[0].shape)
                print("prompt.shape:", prompts[0].shape)

            # 模型预测
            _, pred_masks, _, _ = tented_model(images, prompts)

            for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                pred_bin = (pred_mask >= 0.5).float()

                # 保存每一张预测掩码图
                pred_np = pred_bin[0].cpu().numpy().astype(np.uint8) * 255
                save_path = os.path.join(pred_mask_dir, f"iter{iter}_img{i}_pred.png")
                Image.fromarray(pred_np).save(save_path)

                # 准备 Ground Truth 掩码
                if gt_mask.max() > 1:
                    gt_bin = (gt_mask == 255).float()
                else:
                    gt_bin = gt_mask.float()
                gt_bin = gt_bin.to(fabric.device)

                # Dice
                batch_dice = calculate_dice_torch(gt_bin, pred_bin).mean().item()

                # IoU 和 F1
                pred_mask = pred_mask.to(fabric.device)
                gt_mask = gt_mask.to(fabric.device)
                batch_stats = smp.metrics.get_stats(pred_mask, gt_mask.int(), mode='binary', threshold=0.5)
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                
                csv_dict = {
                "Name": iter,
                "Prompt": cfg.prompt,
                "Mean Dice": f"{batch_dice:.4f}",
                "Mean IoU": f"{batch_iou:.4f}",
                "Mean F1": f"{batch_f1:.4f}",
                "iters": iters
                }
                # 每条都写入csv
                write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
                
                # 累加
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                dice_scores.update(batch_dice, num_images)
                
                if iter == 3:
                    exit()

            fabric.print(
                f'test: [{iters}] - [{iter}/{len(test_dataloader)}]: -- Mean Dice: [{dice_scores.avg:.4f}] Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

            torch.cuda.empty_cache()

    fabric.print(f'Test [{iters}]:  Mean Dice: [{dice_scores.avg:.4f}]  -- Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    # ✅ 写入 CSV
    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "Mean IoU": f"{ious.avg:.4f}",
        "Mean F1": f"{f1_scores.avg:.4f}",
        "iters": iters
    }
    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)

    return ious.avg, f1_scores.avg, dice_scores.avg
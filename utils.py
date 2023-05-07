# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 20:57
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import glob
import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io

cmap = plt.cm.jet


def parse_command():
    modality_names = ["rgb", "rgbd", "d"]

    import argparse

    parser = argparse.ArgumentParser(description="FCRN")
    parser.add_argument("--decoder", default="upproj", type=str)
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)",
    )
    parser.add_argument(
        "-b", "--batch-size", default=16, type=int, help="mini-batch size (default: 4)"
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 15)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.00001,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 0.00001)",
    )
    parser.add_argument(
        "--lr_patience",
        default=2,
        type=int,
        help="Patience of LR scheduler. " "See documentation of ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=0.0005,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 10)",
    )
    parser.add_argument("--dataset", type=str, default="nyu")
    parser.add_argument(
        "--manual_seed", default=1, type=int, help="Manually set random seed"
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 1)",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit to a small subset of data, meant for debugging",
    )
    args = parser.parse_args()
    return args


def get_output_directory(args):
    if args.resume:
        return os.path.dirname(args.resume)
    else:
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = os.path.join(save_dir_root, "result", args.decoder)
        runs = sorted(glob.glob(os.path.join(save_dir_root, "run_*")))
        run_id = int(runs[-1].split("_")[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, "run_" + str(run_id))
        return save_dir


# 保存检查点
def save_checkpoint(state, is_best, epoch, output_directory):
    # only save best model
    if is_best:
        best_filename = os.path.join(output_directory, "model_best.pth.tar")
        torch.save(state, best_filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row_colorization(rgb, grayscale, colorized):
    rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))  # H, W, C
    grayscale = 255 * np.transpose(np.squeeze(grayscale.data.cpu().numpy()), (1, 2, 0))  # H, W, C
    img_merge = np.hstack([rgb, grayscale, colorized])

    return img_merge


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(
        np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu)
    )
    d_max = max(
        np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu)
    )
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype("uint8"))
    img_merge.save(filename)


def get_sample_imgs(sample, loader):
    input, target = sample
    input, target = loader.dataset.inv_preprocess(input[0], target[0])
    input, target = loader.dataset.inv_val_transform(input, target)
    colored_target = colored_depthmap(target)
    return input, colored_target


def to_rgb(ab_input, rgb_true):
    '''Show/save rgb image from grayscale and ab channels
        Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    # convert to lab
    rgb = 255 * np.transpose(np.squeeze(rgb_true.cpu().numpy()), (1, 2, 0))
    lab = rgb2lab(rgb.astype(np.uint8))
    # get lightness channel
    lightness = lab[:, :, 0:1]
    # tranpose back to pytorch order and cast to pytorch
    lightness_tensor = torch.from_numpy(lightness.transpose((2, 0, 1)))

    color_image = torch.cat((lightness_tensor, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0)) 
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
    color_image = 255 * lab2rgb(color_image.astype(np.float64))
    return color_image

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F
import sys
import pdb


from utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs


def generate_frame_indices(crt_idx,
                           max_frame_num,
                           num_frames,
                           padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_txt(txts, keys, filename_tmpl, name):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    if name=='animeFaces-8':
        # animeFaces
        label_dict_train = {"C_dark_1_d70_wb": 0, "C_dark_1_d100_wall": 1, "C_dark_2_d70_wall": 2, "C_dark_2_d70_wb": 3, 
                            "C_dark_2_d100_wall": 4, "C_dark_2_d100_wb": 5, "C_day_1_d70_wall": 6, "C_day_1_d70_wb": 7}
        
    elif name=='Supermodel-8':
        # Supermodel
        label_dict_train = {"C_dark_1_d70_wb": 0, "C_dark_1_d100_wall": 1, "C_dark_2_d70_wall": 2, "C_dark_2_d70_wb": 3, 
                            "C_dark_2_d100_wall": 4, "C_dark_2_d100_wb": 5, "C_day_1_d70_wb": 6, "C_day_2_d70_wall": 7}
    elif name=='MNIST-8':
        # MNIST
        label_dict_train = {"C_dark_1_d70_wall": 0, "C_dark_1_d70_wb": 1, "C_dark_1_d100_wall": 2, 
                            "C_dark_1_d100_wb": 3, "C_dark_2_d70_wall": 4, "C_dark_2_d70_wb": 5, "C_dark_2_d100_wall": 6,
                            "C_dark_2_d100_wb": 7}
    elif name=='STL10-8':
        # STL10
        label_dict_train = {"C_dark_1_d70_wall_occluder": 0, "C_dark_1_d70_wb_occluder": 1, "C_dark_1_d100_wall_occluder": 2, 
                            "C_dark_1_d100_wb_occluder": 3, "C_dark_2_d70_wall_occluder": 4, "C_dark_2_d70_wb_occluder": 5, "C_dark_2_d100_wall_occluder": 6,
                            "C_dark_2_d100_wb_occluder": 7}

    assert len(txts) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(txts)}')
    assert len(keys) == 3, (
        'The len of keys should be 3 with [input_key, gt_key, label]. '
        f'But got {len(keys)}')
    input_txt, gt_txt = txts
    with open(input_txt, "r") as f:
        contents = f.readlines()
        input_paths = [i.strip() for i in contents]
        
    with open(gt_txt, "r") as f:
        contents = f.readlines()
        gt_paths = [i.strip() for i in contents]

    input_key, gt_key, label_key = keys

    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx] 
        input_path = input_paths[idx] 
        input_path = osp.join(input_path)
        gt_path = osp.join(gt_path)
        name = input_path.split('/')[3]
        if(name!='stl10_genera' and name!='test'):
            label = label_dict_train[input_path.split('/')[3]]
        elif((input_path.split('/')[4][-3:])=="val"):
            label = label_dict_train[input_path.split('/')[4][:-4]] # val
        else:
            label = label_dict_train[input_path.split('/')[4][:-5]] # test
        paths.append(
            dict([(f'{input_key}_path', input_path),
                (f'{gt_key}_path', gt_path),
                (f'{label_key}', label)]))
    return paths


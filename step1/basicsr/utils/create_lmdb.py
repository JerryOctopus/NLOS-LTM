# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
from os import path as osp
import sys
import pdb
from utils.misc import scandir
from utils.lmdb_util import make_lmdb_from_imgs



def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reds():

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_gopro():
    
    folder_path = './datasets/AONLOS/STL10/test/C_dark_1_d100_wall_occluder_new_val'
    lmdb_path = './datasets/STL10/val/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/AONLOS/STL10/test/gt_val'
    lmdb_path = './datasets/STL10/val/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/AONLOS/STL10/test/C_dark_1_d100_wall_occluder_new_test'
    lmdb_path = './datasets/STL10/test/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/AONLOS/STL10/test/gt_test'
    lmdb_path = './datasets/STL10/test/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_SIDD():
    folder_path = './datasets/SIDD/train/input_crops'
    lmdb_path = './datasets/SIDD/train/input_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/train/gt_crops'
    lmdb_path = './datasets/SIDD/train/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


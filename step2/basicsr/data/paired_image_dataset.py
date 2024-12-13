# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torchvision.transforms as transforms
from PIL import Image

from data.data_util import paired_paths_from_txt
from utils import FileClient, imfrombytes, imfrompath, img2tensor, padding, resize
import cv2
import pdb


class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'txt': 
            self.paths = paired_paths_from_txt([self.lq_folder, self.gt_folder], ['lq', 'gt', 'label'],
                self.filename_tmpl, opt['name'])
        
        transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        
        if self.io_backend_opt['type'] != 'txt':
            if self.file_client is None:
                self.file_client = FileClient(
                    self.io_backend_opt.pop('type'), **self.io_backend_opt)


        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path'] 
        if self.io_backend_opt['type'] != 'txt':
            img_bytes = self.file_client.get(gt_path, 'gt')
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))
        else:
            try:
                img_gt = imfrompath(gt_path, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        label = self.paths[index]['label']
        # print(', lq path', lq_path)
        if self.io_backend_opt['type'] != 'txt':
            img_bytes = self.file_client.get(lq_path, 'lq')
            try:
                img_lq = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("lq path {} not working".format(lq_path))
        else:
            try:
                img_lq = imfrompath(lq_path, float32=True)
            except:
                raise Exception("lq path {} not working".format(lq_path))


        # augmentation for training
        #if self.opt['phase'] == 'train':
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_size = self.opt['gt_size']

        gt_path = self.paths[index]['gt_path'] 
        img_gt = Image.open(gt_path).convert('RGB')
        img_gt = img_gt.resize((256, 256), Image.BICUBIC)
        lq_path = self.paths[index]['lq_path']
        img_lq = Image.open(lq_path).convert('RGB')
        img_lq = img_lq.resize((256, 256), Image.BICUBIC)

        img_lq = self.transform(img_lq)
        img_gt = self.transform(img_gt)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'label': label
        }

    def __len__(self):
        return len(self.paths)
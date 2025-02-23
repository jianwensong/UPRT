from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, mat2tensor
import os
import math
import numpy as np
import scipy.io

class PairedFringeDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

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
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedFringeDataset, self).__init__()
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

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = os.listdir(self.gt_folder+'/iPattern')
            #self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            nums_gt = len(self.paths)
        self.nums = nums_gt

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_PSin = scipy.io.loadmat(os.path.join(self.gt_folder, 'PSin',self.paths[index]))
        gt_PCos = scipy.io.loadmat(os.path.join(self.gt_folder, 'PCos',self.paths[index]))

        iPattern = scipy.io.loadmat(os.path.join(self.lq_folder, 'iPattern',self.paths[index]))
        # gt_PSin = scipy.io.loadmat(os.path.join(self.gt_folder, 'PSin','{:04}.mat'.format(index+1)))
        # gt_PCos = scipy.io.loadmat(os.path.join(self.gt_folder, 'PCos','{:04}.mat'.format(index+1)))

        # iPattern = scipy.io.loadmat(os.path.join(self.lq_folder, 'iPattern','{:04}.mat'.format(index+1)))
        
        gt_PSin = np.expand_dims(np.array(gt_PSin['PSin']),axis=-1).astype(np.float32)/127.5
        gt_PCos = np.expand_dims(np.array(gt_PCos['PCos']),axis=-1).astype(np.float32)/127.5
        img_lq = np.expand_dims(np.array(iPattern['iPattern']),axis=-1).astype(np.float32)/255.

        img_gt = np.concatenate([gt_PSin, gt_PCos], axis=-1)

        if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
            gt_size_h = int(self.opt['gt_size_h'])
            gt_size_w = int(self.opt['gt_size_w'])
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, 1,
                                                'gt_path_L_and_R')
    
        # numpy to tensor
        img_gt, img_lq = mat2tensor([img_gt, img_lq], float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(index),
            'gt_path': str(index)
        }

    def __len__(self):
        return self.nums


class PairedFringeDatasetTest(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

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
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedFringeDatasetTest, self).__init__()
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

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = os.listdir(self.gt_folder+'/iPattern')
            #self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            nums_gt = len(self.paths)
        self.nums = nums_gt

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_PSin = scipy.io.loadmat(os.path.join(self.gt_folder, 'PSin',self.paths[index]))
        gt_PCos = scipy.io.loadmat(os.path.join(self.gt_folder, 'PCos',self.paths[index]))
        order = scipy.io.loadmat(os.path.join(self.gt_folder, 'Fringe',self.paths[index]))
        iPattern = scipy.io.loadmat(os.path.join(self.lq_folder, 'iPattern',self.paths[index]))
        
        gt_PSin = np.expand_dims(np.array(gt_PSin['PSin']),axis=-1).astype(np.float32)/127.5
        gt_PCos = np.expand_dims(np.array(gt_PCos['PCos']),axis=-1).astype(np.float32)/127.5
        order = np.expand_dims(np.array(order['Order']),axis=-1).astype(np.float32)
        img_lq = np.expand_dims(np.array(iPattern['iPattern']),axis=-1).astype(np.float32)/255.

        img_gt = np.concatenate([gt_PSin, gt_PCos, order], axis=-1)

        if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
            gt_size_h = int(self.opt['gt_size_h'])
            gt_size_w = int(self.opt['gt_size_w'])
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, 1,
                                                'gt_path_L_and_R')
    
        # numpy to tensor
        img_gt, img_lq = mat2tensor([img_gt, img_lq], float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(index),
            'gt_path': str(index)
        }

    def __len__(self):
        return self.nums
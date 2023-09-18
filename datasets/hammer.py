import os
import warnings

import numpy as np
import cv2
import json
import h5py
from . import BaseDataset

import random

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore", category=UserWarning)

class HammerDataset(BaseDataset):
    def __init__(self, args, mode):
        """object initializer

        Args:
            args (obj): execution arguments, the used arguments are:
                        * dir_data: the path to the root of the dataset
                        * data_txt: the text file containing a list of paths to each data, there should be a "DATA_ROOT" placeholder to substitute the data root path into
                        * path_to_vd: the path to the npy file containing the viewing direction data
                        * use_norm: boolean indicating if the normals are used
                        * use_pol: boolean indicating if the polarization data are used


            mode (str): can be either 'train' or 'val' or 'test', which alters the behavior during data loading
        """
        super(HammerDataset, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # -- the camera intrinsic parameters --
        self.K = torch.Tensor([7.067553100585937500e+02, 7.075133056640625000e+02, 5.456326819328060083e+02, 3.899299663507044897e+02])

        # -- the data files --
        with open(args.data_txt.replace("MODE", mode), "r") as file:
            files_names = file.read().split("\n")[:-1]

        self.rgb_files = [s.replace("DATA_ROOT", args.dir_data) for s in files_names] # note that the original paths in the path list is pointing to the rgb images
        
        PERCENTAGE = 1
        if PERCENTAGE < (1-1e-6):
            random.shuffle(self.rgb_files)
            self.rgb_files = self.rgb_files[:int(len(self.rgb_files) * PERCENTAGE)]

        self.sparse_depth_d435_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_d435") for s in files_names]
        self.sparse_depth_l515_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_l515") for s in files_names]
        self.sparse_depth_itof_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_tof") for s in files_names]
        self.gt_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "_gt") for s in files_names]
        if self.args.use_pol:    
            self.pol_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "pol_npy") for s in files_names] # note that the polarizatins are stored as npy files
        if self.args.use_norm:
            self.norm_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "norm").replace(".png", ".npy") for s in files_names] # note that the normals are stored as npy files

    def __len__(self):
        return len(self.rgb_files) * 3
        # return 60

    def __getitem__(self, idx):
        """return data item

        Args:
            idx (int): the index of the date element within the mini-batch

        Return:
            output (dict): the output dictionary of four elements, which are
                        * 'rgb': the pytorch tensor of the RGB 3-channel image data
                        * 'dep': the pytorch tensor of the 1-channel sparse depth (i.e. to be completed)
                        * 'gt': the pytorch tensor of the 1-channel groundtruth depth
                        * 'K': the pytorch tensor of the camera intrinsic matrix, defined to be [fx, fy, cx, cy]
                        * 'mask': the mask of invalid data in the groundtruth
                        * 'pol': [IF use_pol IS TRUE] the pytorch tensor of the 7-channel polarization representation, otherwise all-zero
                        * 'norm': [IF use_norm IS TRUE] the pytorch tensor of the 3-channel normal map, otherwise all-zero
        """
        true_idx = idx // 3
        orig_idx = idx
        idx = true_idx

        def np2tensor(sample):
            # HWC -> CHW
            sample_tensor = torch.from_numpy(sample.copy().astype(np.float32)).permute(2,0,1)
            return sample_tensor
        
        # -- prepare sparse depth --
        sparse_depth_file = None
        depth_type = orig_idx % 3

        if depth_type == 0:
            sparse_depth_file = self.sparse_depth_d435_files[idx]
        elif depth_type == 1:
            sparse_depth_file = self.sparse_depth_l515_files[idx]
        elif depth_type == 2:
            sparse_depth_file = self.sparse_depth_itof_files[idx]

        sparse_depth = cv2.imread(sparse_depth_file, -1)[:,:,None] # (H, W, 1)
        sparse_depth = sparse_depth[::4,::4,...]
        sparse_depth = np2tensor(sparse_depth) # (1, H, W)

        # -- prepare gt depth --
        gt = cv2.imread(self.gt_files[idx], -1)[:,:,None] # (H, W, 1)
        gt = gt[::4,::4,...]
        gt_clone = np.copy(gt)
        gt = np2tensor(gt) # (1, H, W)

        # -- prepare rgb --
        rgb = cv2.imread(self.rgb_files[idx]) # (H, W, 3)
        rgb = rgb[::4,::4,...]
        rgb = np2tensor(rgb) # (3, H, W)

        # -- prepare normals --
        if self.args.use_norm:
            norm = np.load(self.norm_files[idx]) # (H, W, 3)
            norm = norm[::4,::4,...]
            norm = np2tensor(norm) # (3, H, W)

        # -- prepare intrinsics --
        K = self.K.clone()

        # -- prepare mask, which masks out invalid pixels in the groundtruth --
        mask = np.ones_like(gt_clone) # (H, W, 1)
        mask[gt_clone<1e-3] = 0
        mask = np2tensor(mask) # (1, H, W)

        # -- prepare polarization representation -- 
        # QUESTION: What is the definition of the data contained in the polarization npy files?
        if self.args.use_pol:
            pol = np.load(self.pol_files[idx])
            pol = pol[::4,::4,...]
            phi_encode = np.concatenate([np.cos(2 * pol[..., 3:6]), np.sin(2 * pol[..., 3:6])], axis=2)
            pol = np.concatenate([pol[..., 0:3], phi_encode, pol[..., 6:9], pol[..., 9:12]], axis=2)

        # -- apply data augmentation --
        if self.mode == "train":
            t_rgb = T.Compose([
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            rgb = t_rgb(rgb)
            
        # -- return data --
        # print("--> RGB size {}".format(rgb.shape))
        # print("--> Sparse depth size {}".format(sparse_depth.shape))
        # print("--> Groundtruth size {}".format(gt.shape))
        # print("--> Mask size {}".format(mask.shape))
        output = {'rgb': rgb, \
                    'dep': sparse_depth, \
                    'gt': gt, \
                    'K': K, \
                    'net_mask': mask}
        
        if self.args.use_pol:
            output['pol'] = pol

        if self.args.use_norm:
            output['norm'] = norm

        return output
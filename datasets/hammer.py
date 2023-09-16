import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import random
import warnings


class HammerDataset(Dataset):

    def __init__(self, args, paths_list_txt, path_to_viewing_dir):
        super(HammerDataset, self).__init__()
        
        self.data_root = args.data_root
        self.is_train = args.is_train
        self.cropped_power = args.cropped_power
        self.new_H = self.new_W = 2 ** self.cropped_power
        self.use_min_max_norm = args.use_min_max_norm
        # read data paths
        files = open(paths_list_txt, "r").read().split("\n")[:-1] # assume the last line is an empty line, so discard
        files = [f.replace("DATA_ROOT", self.data_root) for f in files]
            
        orig_len = len(files)
            
        print("--> Using {} percent of original data ({}), that is {}".format(self.data_percentage, orig_len, len(files)))
        
        # get paths to all required data
        self.polarization_paths = [f.replace("rgb", "pol_npy").replace('.png', '.npy') for f in files]
        self.norm_paths = [f.replace("rgb", "norm").replace('png', 'npy') for f in files]
        self.depth_paths = [f.replace("rgb", "depth_d435") for f in files]
        self.gt_depth_paths = [f.replace("rgb", "_gt") for f in files]
        self.image_paths = files
        self.viewing_dir = np.load(path_to_viewing_dir)
        
    def __getitem__(self, index):
        
        self.rand_int1 = np.random.randint(0, 512-1)
        self.rand_int2 = np.random.randint(0, 512-1)
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        # read raw data, all as numpy arrays
        polarization = np.load(self.polarization_paths[index])
        phi_encode = np.concatenate([np.cos(2 * polarization[..., 3:6]), np.sin(2 * polarization[..., 3:6])], axis=2)
        net_in = np.concatenate([polarization[..., 0:3], phi_encode, 
            polarization[..., 6:9], polarization[..., 9:12]], axis=2)

        # norm = cv2.imread(self.norm_paths[index], -1) # BGR, i.e. zyx, 0-(2^16-1) (mapping from -1 to 1)
        norm = np.load(self.norm_paths[index])
        depth = cv2.imread(self.depth_paths[index], -1) # in mm
        gt_depth = cv2.imread(self.gt_depth_paths[index], -1)
        # mask = np.ones_like(depth) if not os.path.exists(self.mask_paths[index]) else cv2.imread(self.mask_paths[index], -1) # mask may not exist
        mask = np.ones_like(gt_depth)
        mask[gt_depth<1e-3] = 0
        image = cv2.imread(self.image_paths[index]) # read as rgb image (but is grayscale)
        
        # data_list = polarization, norm, depth, gt_depth, mask, image, self.viewing_dir
        data_list = [net_in, norm, depth, gt_depth, mask, image, self.viewing_dir]
        
        # process the raw data
        data_list[1] = data_list[1].astype(np.float32)
        vecotr_magnitudes = np.sqrt(np.sum(np.square(data_list[1]), axis=2))
        data_list[4][vecotr_magnitudes<=0.3] = 0 # filter outs vectors with too small magnitudes (most likely invalid)
        data_list[1][data_list[3]==1] = data_list[1][data_list[3]==1] / np.linalg.norm(data_list[1], axis=2)[data_list[3]==1][:,np.newaxis] # perform normalization to constrain the vectors to have magnitudes of 1
        data_list[1][data_list[3]==0] = 0 # mask out invalid data
        data_list[2][data_list[2]<10] = 0 # get rid of anything having distance smaller than 1cm (10mm)
        data_list[5] = data_list[5] / 255.0 # to [0, 1]
        data_list[2] = data_list[2][:, :, np.newaxis]
        data_list[3] = data_list[3][:, :, np.newaxis]
        data_list[4] = data_list[4][:, :, np.newaxis]

        # Resize
        data_list = [self.resize(i, self.new_W, self.new_H, mode='slice') for i in data_list]

        if self.use_min_max_norm:
            data_list[2] = data_list[2] / data_list[2].max()
        
        def np2tensor(sample):
            # HWC -> CHW
            sample_tensor = torch.from_numpy(sample.copy().astype(np.float32)).permute(2,0,1)
            return sample_tensor
            
        data_list = [np2tensor(sample) for sample in data_list]
    
            
        if data_list[3].max() == 0:
            # if we encounter all zero mask (possible since we do random cropping on the original data)
            # we randomly set some masks to be 1
            rand_x = np.random.randint(0, 511-50)
            rand_y = np.random.randint(0, 511-50)
            data_list[3][:, rand_y:rand_y+50, rand_x:rand_x+50] = 1
            # raise Exception("Received all zero mask! Path: {}".format(self.mask_paths[index]))
        return data_list

    def resize(self, img, target_w, target_h, mode='bilinear'):
        '''
            img: h, w, c, c in BGR
            w: target width
            h: target height
        '''
        if mode == 'bilinear':
            ch = img.shape[2]
            img = cv2.resize(img, (target_w, target_h), cv2.INTER_LINEAR)
            if ch == 1:
                img = np.expand_dims(img, axis=2)
        elif mode == 'slice':
            ori_w = img.shape[1]
            ori_h = img.shape[0]
            w_interval = np.floor(ori_w/target_w).astype(int)
            h_interval = np.floor(ori_h/target_h).astype(int)

            img = img[::h_interval, ::w_interval, :]
            new_h, new_w, _ = img.shape
            img = img[(new_h-target_h)//2:target_h+(new_h-target_h)//2, 
                      (new_w-target_w)//2:target_w+(new_w-target_w)//2, :]

        return img
        
    def __len__(self):
        return len(self.polarization_paths)
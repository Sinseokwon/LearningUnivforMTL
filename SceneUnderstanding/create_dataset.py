import os
import cv2
import random
import torch
import fnmatch

import numpy as np
import panoptic_parts as pp
import torch.utils.data as data
import matplotlib.pylab as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from PIL import Image


class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:

            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:

            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict['im'].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  
        flip_prop = random.random()

        for task in data_dict:
            if len(data_dict[task].shape) == 2: 
                data_dict[task] = data_dict[task].unsqueeze(0)

            if task in ['im', 'noise']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.BILINEAR)
            elif task in ['normal', 'depth', 'seg', 'part_seg', 'disp']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.NEAREST)

            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(self.crop_size[0] - resized_size[0], 0)
                if task in ['im']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       padding_mode='reflect')
                elif task in ['seg', 'part_seg', 'disp']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=-1, padding_mode='constant') 
                elif task in ['normal', 'depth', 'noise']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=0, padding_mode='constant') 

            if i + j + h + w == 0:  
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == 'normal':
                    data_dict[task][0, :, :] = - data_dict[task][0, :, :]

            if task == 'depth':
                data_dict[task] = data_dict[task] / sc

            if task == 'disp':  
                data_dict[task] = data_dict[task] * sc

            if task in ['seg', 'part_seg']:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict


class NYUv2(data.Dataset):

    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation


        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'


        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):

        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
        noise = self.noise[index].float()

        data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal, 'noise': noise}

        if self.augmentation:
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1. 
        return im, data_dict

    def __len__(self):
        return self.data_len


class CityScapes(data.Dataset):

    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.png'))
        self.noise = torch.rand(self.data_len, 1, 256, 256) if self.train else torch.rand(self.data_len, 1, 256, 512)

    def __getitem__(self, index):
        image = torch.from_numpy(np.moveaxis(plt.imread(self.data_path + '/image/{:d}.png'.format(index)), -1, 0)).float()
        disparity = cv2.imread(self.data_path + '/depth/{:d}.png'.format(index), cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity = torch.from_numpy(self.map_disparity(disparity)).unsqueeze(0).float()
        seg = np.array(Image.open(self.data_path + '/seg/{:d}.png'.format(index)), dtype=float)
        seg = torch.from_numpy(self.map_seg_label(seg)).long()
        part_seg = np.array(Image.open(self.data_path + '/part_seg/{:d}.tif'.format(index)))
        part_seg = torch.from_numpy(self.map_part_seg_label(part_seg)).long()
        noise = self.noise[index].float()

        data_dict = {'im': image, 'seg': seg, 'part_seg': part_seg, 'disp': disparity, 'noise': noise}

        if self.augmentation:
            data_dict = DataTransform(crop_size=[256, 256], scales=[1.0])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  
        return im, data_dict

    def map_seg_label(self, mask):
        mask_map = np.zeros_like(mask)
        mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = -1
        mask_map[np.isin(mask, [7])] = 0
        mask_map[np.isin(mask, [8])] = 1
        mask_map[np.isin(mask, [11])] = 2
        mask_map[np.isin(mask, [12])] = 3
        mask_map[np.isin(mask, [13])] = 4
        mask_map[np.isin(mask, [17])] = 5
        mask_map[np.isin(mask, [19])] = 6
        mask_map[np.isin(mask, [20])] = 7
        mask_map[np.isin(mask, [21])] = 8
        mask_map[np.isin(mask, [22])] = 9
        mask_map[np.isin(mask, [23])] = 10
        mask_map[np.isin(mask, [24])] = 11
        mask_map[np.isin(mask, [25])] = 12
        mask_map[np.isin(mask, [26])] = 13
        mask_map[np.isin(mask, [27])] = 14
        mask_map[np.isin(mask, [28])] = 15
        mask_map[np.isin(mask, [31])] = 16
        mask_map[np.isin(mask, [32])] = 17
        mask_map[np.isin(mask, [33])] = 18
        return mask_map

    def map_part_seg_label(self, mask):

        mask = pp.decode_uids(mask, return_sids_pids=True)[-1]
        mask_map = np.zeros_like(mask)  # background
        mask_map[np.isin(mask, [2401, 2501])] = 1    # human/rider torso
        mask_map[np.isin(mask, [2402, 2502])] = 2    # human/rider head
        mask_map[np.isin(mask, [2403, 2503])] = 3    # human/rider arms
        mask_map[np.isin(mask, [2404, 2504])] = 4    # human/rider legs
        mask_map[np.isin(mask, [2601, 2701, 2801])] = 5  # car/truck/bus windows
        mask_map[np.isin(mask, [2602, 2702, 2802])] = 6  # car/truck/bus wheels
        mask_map[np.isin(mask, [2603, 2703, 2803])] = 7  # car/truck/bus lights
        mask_map[np.isin(mask, [2604, 2704, 2804])] = 8  # car/truck/bus license_plate
        mask_map[np.isin(mask, [2605, 2705, 2805])] = 9  # car/truck/bus chassis
        return mask_map

    def map_disparity(self, disparity):
        disparity[disparity == 0] = -1
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        return disparity

    def __len__(self):
        return self.data_len

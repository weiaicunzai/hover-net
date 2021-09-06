import glob
import logging
import math
import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import sys
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeFileList
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage import color

import convert_format
from . import base


####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.

    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img

    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]
    #assert im_h == im_w

    #print('step_size', step_size)
    # step_size 80
    # print(img.shape, msk_size, step_size) (1792, 1792, 3) 80 80
    last_h, nr_step_h = get_last_steps(im_h, msk_size, step_size)
    last_w, nr_step_w = get_last_steps(im_w, msk_size, step_size)

    # print(last_h, last_w) 1840


    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    sub_patches = []
        # generating subpatches from orginal
    #print('img.shape', img.shape)

    for row in range(0, last_h, step_size):
        for col in range (0, last_w, step_size):
                win = img[row:row+win_size,
                        col:col+win_size]
                sub_patches.append(win)

    #print(111111111111, count)
    #print(len(sub_patches))
    # generating subpatches index from orginal
    #coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    #coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    #row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    #col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    #coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    #row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    #coord_y = coord_y.flatten()
    #coord_x = coord_x.flatten()
    #row_idx = row_idx.flatten()
    #col_idx = col_idx.flatten()
    #
    #print(coord_x, coord_x.shape)
    #print(coord_y, coord_y.shape)
    #print(row_idx, row_idx.shape)
    #print(col_idx, col_idx.shape)
    #patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    #print(patch_info, patch_info.shape)
    #import sys; sys.exit()
    #if not return_src_top_corner:
    return sub_patches, nr_step_h, nr_step_w, im_h, im_w
    #else:
    #    return img, patch_info, [padt, padl]


class ExtendedCRC(data.Dataset):
    def __init__(self, path, win_size, msk_size, return_src_top_corner):
        self.image_names = []

        pp = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/json/'
        #json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        #print(json_lists[3])
        #sys.exit()
        #print(len(json_lists))
        #sys.exit()
        for path in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            p = os.path.basename(path).split('.')[0]
            #print(p, json_lists)
            if p not in json_lists:
                #print(path)


                self.image_names.append(path)

        #print(len(self.image_names))
        #sys.exit()
        self.win_size = win_size
        self.msk_size = msk_size
        self.return_src_top_corner = return_src_top_corner

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        image = cv2.imread(path, -1)
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        src_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #return sub_patches, nr_step_h, nr_step_w, im_h, im_w
        #sub_images, nr_step, src_shape
        sub_images, nr_step_h, nr_step_w, im_h, im_w = _prepare_patching(src_image, self.win_size, self.msk_size, self.return_src_top_corner)
        #sub_images = []
        #for i in range(36):
            #sub_images.append(str(i) + path)

        return  np.array(sub_images), nr_step_h, nr_step_w, im_h, im_w, path
        #return  sub_images, path

#class Prostate(data.Dataset):
#    def __init__(self, path, win_)

class BACH(data.Dataset):
    def __init__(self, path, win_size, msk_size, return_src_top_corner):
        self.image_names = []

        #pp = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/json/'
        ##json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        #json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        #print(json_lists[3])
        #sys.exit()
        #print(len(json_lists))
        #sys.exit()
        #for path in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
        for img_fp in glob.iglob(os.path.join(path, '**', '*.tif'), recursive=True):
            #p = os.path.basename(img_fp).split('.')[0]
            #print(p, json_lists)
            #if p not in json_lists:
                #print(path)
            self.image_names.append(img_fp)

        #print(len(self.image_names))
        #sys.exit()
        self.win_size = win_size
        self.msk_size = msk_size
        self.return_src_top_corner = return_src_top_corner

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        image = cv2.imread(path, -1)
        #image = cv2.resize(image, (0, 0), fx=2, fy=2)
        src_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')
        #src_image = cv2.resize(src_image, (0, 0), fx=2, fy=2)
        #print(src_image.shape, src_image.dtype, src_image.max())
        #print(src_image.shape, src_image.dtype, src_image.astype('uint8').max())
        #sub_images, nr_step, src_shape = _prepare_patching(src_image, self.win_size, self.msk_size, self.return_src_top_corner)
        #return  np.array(sub_images), nr_step_h, nr_step_w, im_h, im_w, path

        sub_images, nr_step_h, nr_step_w, im_h, im_w = _prepare_patching(src_image, self.win_size, self.msk_size, self.return_src_top_corner)

        #return  np.array(sub_images), nr_step, src_shape, path
        return  np.array(sub_images), nr_step_h, nr_step_w, im_h, im_w, path

class Prostate(data.Dataset):
    def __init__(self, path, win_size, msk_size, return_src_top_corner):
        self.image_names = []

        #pp = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/json/'
        ##json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        #json_lists = set([i.split('.')[0] for i in os.listdir(pp)])
        #print(json_lists[3])
        #sys.exit()
        #print(len(json_lists))
        #sys.exit()
        #for path in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
        for img_fp in glob.iglob(os.path.join(path, '**', '*.jpg'), recursive=True):
            #p = os.path.basename(img_fp).split('.')[0]
            #print(p, json_lists)
            #if p not in json_lists:
                #print(path)
            self.image_names.append(img_fp)

        #print(len(self.image_names))
        #sys.exit()
        self.win_size = win_size
        self.msk_size = msk_size
        self.return_src_top_corner = return_src_top_corner

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        image = cv2.imread(path, -1)
        #image = cv2.resize(image, (0, 0), fx=2, fy=2)
        src_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sub_images, nr_step, src_shape = _prepare_patching(src_image, self.win_size, self.msk_size, self.return_src_top_corner)

        return  np.array(sub_images), nr_step, src_shape, path

class InferManager(base.InferManager):
    """Run inference on tiles."""


    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)

        def assemble_pred_maps(pred_map, nr_step_h, nr_step_w, im_h, im_w):
            output_patch_shape = np.squeeze(pred_map[0]).shape
            ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

                    #### Assemble back into full image
            pred_map = np.squeeze(np.array(pred_map))
            pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
            pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
            pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
            pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

            return pred_map

        #dataset = ExtendedCRC(
        #dataset = Prostate(
        dataset = BACH(
            path=self.input_dir,
            win_size=self.patch_input_shape,
            msk_size=self.patch_output_shape,
            return_src_top_corner=True)
        # * apply neural net on cached data
        #dataset = SerializeFileList(
        #    cache_image_list, cache_patch_info_list, self.patch_input_shape
        #)

        print('dataset size', len(dataset))
        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=2,
            #batch_size=3,
            drop_last=False,
        )

        import time
        start = time.time()
        count = 0
         #read date


        # assume each src_image has the same shape
        #return  np.array(sub_images), nr_step_h, nr_step_w, im_h, im_w, path

        for idx, (sub_images, nr_step_h, nr_step_w, im_h, im_w, path) in enumerate(dataloader):

            #print(self.run_step)
            #if sub_images.shape[0] == self.batch_size:
            #    continue

            #print(sub_images[0].shape)
            #sys.exit()

            num_patches = len(sub_images[0])
            #print(num_patches)
            sub_images = sub_images.reshape(-1, *sub_images.shape[2:])
            #print(sub_images.shape)

            preds = []
            #print(len(sub_images))
            #iter_times = math.ceil(len(sub_images) / self.batch_size)


            prev_start = 0
            print('before', sub_images.shape)
            while True:
                pred = self.run_step(sub_images[prev_start : prev_start+self.batch_size])
                #print(type(pred), pred.shape)
                preds.append(pred)
                prev_start += self.batch_size
                if prev_start >= len(sub_images) - 1:
                    break

            preds = np.concatenate(preds)
            print('after', preds.shape)
            for pred_idx, step_h, step_w, ih, iw, p in zip(range(0, len(preds), num_patches), nr_step_h, nr_step_w, im_h, im_w, path):
                #print(i)
                #print(preds[i:i+num_patches].shape)
                pred = preds[pred_idx : pred_idx+num_patches]
                pred_map = assemble_pred_maps(pred, step_h, step_w, ih, iw)
                #print(pred_map.shape)
                pred_inst, inst_info_dict = self.post_proc_func(pred_map, nr_types=self.nr_types, return_centroids=True)

                base_name = os.path.basename(p)
                image_name = base_name.split('.')[0]
                #save_path = "%s/json/%s.json" % (self.output_dir, image_name)
                save_path = os.path.join(self.output_dir, 'json', '{}.json'.format(image_name))
                os.makedirs(os.path.join(self.output_dir, 'json'), exist_ok=True)
                self.__save_json(save_path, inst_info_dict, None)

            #for pred in preds:

            #### Assemble back into full image
            #output_patch_shape = np.squeeze(pred_map[0]).shape
            #ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

            #        #### Assemble back into full image
            #pred_map = np.squeeze(np.array(pred_map))
            #pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
            #pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
            #            np.transpose(pred_map, [0, 2, 1, 3])
            #pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
            #                             pred_map.shape[2] * pred_map.shape[3], ch))
            #pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size
            #print(pred_map.shape)
            #count += 1
            #if count == 10:
            #    print((time.time() - start) / len(path) / count)
            #    sys.exit()

            #for i in range()
            #for i in range(len(sub_images)):
                #print(i)
            #count += 1
            #if count % 50 == 0:
            #    print(count)
                #print(image.shape)
            #count += 1
            print('iter/total [{}/{}], average speed: {:4f}s'.format(
                    idx * self.batch_size + len(path),
                    len(dataset),
                    (idx * self.batch_size + len(path)) / (time.time() - start))
                )
                #sys.exit()

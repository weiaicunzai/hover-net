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
    assert im_h == im_w

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
    return sub_patches, nr_step_h, im_h
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
        sub_images, nr_step, src_shape = _prepare_patching(src_image, self.win_size, self.msk_size, self.return_src_top_corner)
        #sub_images = []
        #for i in range(36):
            #sub_images.append(str(i) + path)

        return  np.array(sub_images), nr_step, src_shape, path
        #return  sub_images, path

#class Prostate(data.Dataset):
#    def __init__(self, path, win_)

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
####
#def _post_process_patches(
#    post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
#):
#    """Apply post processing to patches.
#
#    Args:
#        post_proc_func: post processing function to use
#        post_proc_kwargs: keyword arguments used in post processing function
#        patch_info: patch data and associated information
#        image_info: input image data and associated information
#        overlay_kwargs: overlay keyword arguments
#
#    """
#    # re-assemble the prediction, sort according to the patch location within the original image
#    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
#    patch_info, patch_data = zip(*patch_info)
#
#    src_shape = image_info["src_shape"]
#    src_image = image_info["src_image"]
#
#    patch_shape = np.squeeze(patch_data[0]).shape
#    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
#    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]
#
#    nr_row = max([x[2] for x in patch_info]) + 1
#    nr_col = max([x[3] for x in patch_info]) + 1
#    pred_map = np.concatenate(patch_data, axis=0)
#    # print(pred_map.shape) # (36, 80, 80, 3)
#    # print(len(patch_info), type(patch_info)) 36
#    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
#    # print(pred_map.shape, 1111) # (6, 6, 80, 80, 3) 1111
#    # print(pred_map.shape, 1) # (6, 6, 80, 80, 3) 1
#    pred_map = np.transpose(pred_map, axes)
#    # print(pred_map.shape, 2) # (6, 80, 6, 80, 3) 2
#    pred_map = np.reshape(
#        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
#    )
#    # print(pred_map.shape, 3) (480, 480, 3) 3
#    # crop back to original shape
#    # print('before:', pred_map.shape)
#    # before: (480, 480, 3)
#    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])
#    # print(pred_map.shape, 4) # (448, 448, 3) 4
#    # print('after:', pred_map.shape)
#    # after: (448, 448, 3)
#
#    # * Implicit protocol
#    # * a prediction map with instance of ID 1-N
#    # * and a dict contain the instance info, access via its ID
#    # * each instance may have type
#    #print(post_proc_func)
#    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)
#    # print(pred_inst.shape, 5) # (448, 448) 5
#
#    overlaid_img = visualize_instances_dict(
#        src_image.copy(), inst_info_dict, **overlay_kwargs
#    )
#
#    return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img


class InferManager(base.InferManager):
    """Run inference on tiles."""


    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        #assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        ## * depend on the number of samples and their size, this may be less efficient
        #patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        #file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        #file_path_list.sort()  # ensure same order
        #assert len(file_path_list) > 0, 'Not Detected Any Files From Path'

        ##rm_n_mkdir(self.output_dir + '/json/')
        ##rm_n_mkdir(self.output_dir + '/mat/')
        ##rm_n_mkdir(self.output_dir + '/overlay/')
        ##if self.save_qupath:
        ##    rm_n_mkdir(self.output_dir + "/qupath/")

        #def proc_callback(results):
        #    """Post processing callback.

        #    Output format is implicit assumption, taken from `_post_process_patches`

        #    """
        #    img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

        #    nuc_val_list = list(inst_info_dict.values())
        #    # need singleton to make matlab happy
        #    nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
        #    nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
        #    nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

        #    mat_dict = {
        #        "inst_map" : pred_inst,
        #        "inst_uid" : nuc_uid_list,
        #        "inst_type": nuc_type_list,
        #        "inst_centroid": nuc_coms_list
        #    }
        #    if self.nr_types is None: # matlab does not have None type array
        #        mat_dict.pop("inst_type", None)

        #    #print(self.save_raw_map, 111)
        #    #import sys; sys.exit()
        #    #if self.save_raw_map:
        #    #    mat_dict["raw_map"] = pred_map
        #    #save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
        #    #sio.savemat(save_path, mat_dict)

        #    #save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
        #    #cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

        #    if self.save_qupath:
        #        nuc_val_list = list(inst_info_dict.values())
        #        nuc_type_list = np.array([v["type"] for v in nuc_val_list])
        #        nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
        #        save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
        #        convert_format.to_qupath(
        #            save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
        #        )

        #    save_path = "%s/json/%s.json" % (self.output_dir, img_name)
        #    self.__save_json(save_path, inst_info_dict, None)
        #    return img_name

        #def detach_items_of_uid(items_list, uid, nr_expected_items):
        #    item_counter = 0
        #    detached_items_list = []
        #    remained_items_list = []
        #    while True:
        #        pinfo, pdata = items_list.pop(0)
        #        pinfo = np.squeeze(pinfo)
        #        if pinfo[-1] == uid:
        #            detached_items_list.append([pinfo, pdata])
        #            item_counter += 1
        #        else:
        #            remained_items_list.append([pinfo, pdata])
        #        if item_counter == nr_expected_items:
        #            break
        #    # do this to ensure the ordering
        #    remained_items_list = remained_items_list + items_list
        #    return detached_items_list, remained_items_list

        #proc_pool = None
        #if self.nr_post_proc_workers > 0:
        #    proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)


        ###########################
        #from .stich import BaseDataset, ImageFolder
        #image_folder = ImageFolder('/data/by/tmp/HGIN/test_can_be_del2')
        #dataset = BaseDataset(image_folder, 224)
        ##file_path_list1 = dataset.image_path_lists
        ##print(len(file_path_list1))
        #json_files = []
        #for j in glob.iglob('/data/by/tmp/hover_net/samples/out/backup_json/**.json'):
        #    bsname = os.path.basename(j)
        #    json_files.append(bsname.split('.')[0])
        #json_files = set(json_files)


        #file_path_list = []
        #count = 0
        #crc_extends_images = []
        #for image, path in dataset:

        #    count += 1
        #    image_base_name = os.path.basename(path)
        #    image_base_prefix = image_base_name.split('.')[0]
        #    if image_base_prefix in json_files:
        #        continue

        #    #print(image_base_prefix)
        #    #import sys; sys.exit()


        ##import sys; sys.exit()
        #    file_path_list.append(path)
        #    crc_extends_images.append(image)
        #    if count != len(dataset):
        #        continue

        #################################################
            #while len(file_path_list) > 0:


            #    hardware_stats = psutil.virtual_memory()
            #    available_ram = getattr(hardware_stats, "available")
            #    available_ram = int(available_ram * self.mem_usage)
            #    # available_ram >> 20 for MB, >> 30 for GB

            #    # TODO: this portion looks clunky but seems hard to detach into separate func

            #    # * caching N-files into memory such that their expected (total) memory usage
            #    # * does not exceed the designated percentage of currently available memory
            #    # * the expected memory is a factor w.r.t original input file size and
            #    # * must be manually provided
            #    file_idx = 0
            #    use_path_list = []
            #    cache_image_list = []
            #    cache_patch_info_list = []
            #    cache_image_info_list = []
            #    #print(file_path_list)
            #    #import sys;sys.exit()
            #    #print(len(file_path_list))
            #    while len(file_path_list) > 0:
            #        file_path = file_path_list.pop(0)
            #        #print(file_path)



            #        img = cv2.imread(file_path)
            #        img = cv2.resize(img, (0, 0), fx=2, fy=2)
            #        #######################################
            #        #img = crc_extends_images.pop()
            #        #img = cv2.resize(img, (0, 0), fx=2, fy=2)
            #        #print(img.shape, file_path, len(file_path_list))
            #        #######################################

            #        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                    src_shape = img.shape
#
#                    #print('before', img.shape, self.patch_input_shape, self.patch_output_shape) # before (448, 448, 3) 270 80
#                    img, patch_info, top_corner = _prepare_patching(
#                        img, self.patch_input_shape, self.patch_output_shape, True
#                    )
#                    # print('after', img.shape) # after (845, 845, 3)
#                    #print(img.shape, patch_info, top_corner)
#                    self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
#                    patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
#                    # ? may be expensive op
#                    patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
#                    patch_info = [np.squeeze(p) for p in patch_info]
#
#                    # * this factor=5 is only applicable for HoVerNet
#                    expected_usage = sys.getsizeof(img) * 5
#                    available_ram -= expected_usage
#                    if available_ram < 0:
#                        break
#
#                    file_idx += 1
#                    # if file_idx == 4: break
#                    use_path_list.append(file_path)
#                    # print(img.shape) # (845, 845, 3)
#                    cache_image_list.append(

#)
#                    cache_patch_info_list.extend(patch_info)
#                    # TODO: refactor to explicit protocol
#                    cache_image_info_list.append([src_shape, len(patch_info), top_corner])
#
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
        dataset = Prostate(
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
            #batch_size=self.batch_size,
            batch_size=3,
            drop_last=False,
        )

        import time
        start = time.time()
        count = 0
         #read date


        # assume each src_image has the same shape
        for idx, (sub_images, nr_step, src_shape, path) in enumerate(dataloader):

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
            for pred_idx, step, im, p in zip(range(0, len(preds), num_patches), nr_step, src_shape, path):
                #print(i)
                #print(preds[i:i+num_patches].shape)
                pred = preds[pred_idx : pred_idx+num_patches]
                pred_map = assemble_pred_maps(pred, step, step, im, im)
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
                    (time.time() - start) / (idx * self.batch_size + len(path))
                ))
                #sys.exit()

        #finish = time.time()
        #print((time.time() - start))






        #pbar = tqdm.tqdm(
        #    desc="Process Patches",
        #    leave=True,
        #    total=int(len(cache_patch_info_list) / self.batch_size) + 1,
        #    ncols=80,
        #    ascii=True,
        #    position=0,
        #)

        #accumulated_patch_output = []
        #for batch_idx, batch_data in enumerate(dataloader):
        #    sample_data_list, sample_info_list = batch_data
        #    # print(11111, sample_data_list.shape) torch.Size([8, 270, 270, 3])
        #    sample_output_list = self.run_step(sample_data_list)
        #    # print(112222, sample_output_list.shape) # 112222 (8, 80, 80, 3)
        #    sample_info_list = sample_info_list.numpy()
        #    curr_batch_size = sample_output_list.shape[0]
        #    #print(sample_output_list.shape) # (7, 80, 80, 3)
        #    sample_output_list = np.split(
        #        sample_output_list, curr_batch_size, axis=0
        #    )
        #    #print(len(sample_output_list), sample_output_list[0].shape) # 8 (1, 80, 80, 3)
        #    sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
        #    sample_output_list = list(zip(sample_info_list, sample_output_list))
        #    accumulated_patch_output.extend(sample_output_list)
        #    pbar.update()
        #pbar.close()

        ## * parallely assemble the processed cache data for each file if possible
        #future_list = []
        #for file_idx, file_path in enumerate(use_path_list):
        #    image_info = cache_image_info_list[file_idx]
        #    file_ouput_data, accumulated_patch_output = detach_items_of_uid(
        #        accumulated_patch_output, file_idx, image_info[1]
        #    )

        #    # * detach this into func and multiproc dispatch it
        #    src_pos = image_info[2]  # src top left corner within padded image
        #    src_image = cache_image_list[file_idx]
        #    src_image = src_image[
        #        src_pos[0] : src_pos[0] + image_info[0][0],
        #        src_pos[1] : src_pos[1] + image_info[0][1],
        #    ]

        #    base_name = pathlib.Path(file_path).stem
        #    file_info = {
        #        "src_shape": image_info[0],
        #        "src_image": src_image,
        #        "name": base_name,
        #    }

        #    post_proc_kwargs = {
        #        "nr_types": self.nr_types,
        #        "return_centroids": True,
        #    }  # dynamicalize this

        #    overlay_kwargs = {
        #        "draw_dot": self.draw_dot,
        #        "type_colour": self.type_info_dict,
        #        "line_thickness": 2,
        #    }
        #    func_args = (
        #        self.post_proc_func,
        #        post_proc_kwargs,
        #        file_ouput_data,
        #        file_info,
        #        overlay_kwargs,
        #    )

        #    # dispatch for parallel post-processing
        #    if proc_pool is not None:
        #        proc_future = proc_pool.submit(_post_process_patches, *func_args)
        #        # ! manually poll future and call callback later as there is no guarantee
        #        # ! that the callback is called from main thread
        #        future_list.append(proc_future)
        #    else:
        #        proc_output = _post_process_patches(*func_args)
        #        proc_callback(proc_output)

        #if proc_pool is not None:
        #    # loop over all to check state a.k.a polling
        #    for future in as_completed(future_list):
        #        # TODO: way to retrieve which file crashed ?
        #        # ! silent crash, cancel all and raise error
        #        if future.exception() is not None:
        #            log_info("Silent Crash")
        #            # ! cancel somehow leads to cascade error later
        #            # ! so just poll it then crash once all future
        #            # ! acquired for now
        #            # for future in future_list:
        #            #     future.cancel()
        #            # break
        #        else:
        #            file_path = proc_callback(future.result())
        #            log_info("Done Assembling %s" % file_path)

        #count = 0
        #file_path_list = []
        #crc_extends_images = []
        #return
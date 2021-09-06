import json
import os
import glob
import random

import cv2
import torch
import numpy as np


def vis(img_fp, json_fp):
    #print(img_fp)
    image = cv2.imread(img_fp)
    #image = img_fp
    #print(json_fp)
    with open(json_fp, 'r') as f:
        res = json.load(f)
        #print(res.keys())
        #print(res['nuc']['2175'])
        print(len(res['nuc'].keys()))
        for node in res['nuc'].keys():
            cen = res['nuc'][node]['centroid']
            cen = [int(c) for c in cen]
            bbox = res['nuc'][node]['bbox']
            image = cv2.circle(image, tuple(cen), 3, (0, 200, 0), cv2.FILLED, 3)

            store_bbox = bbox
            bbox = [b for b in sum(bbox, [])]
            #print('after:')
            if bbox[2] - bbox[0] <= 0:
                print(bbox, store_bbox, '1')
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue
            if bbox[3] - bbox[1] <= 0:
                print(bbox, store_bbox, 2)
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue

            if min(bbox) < 0:
                print(bbox, store_bbox, 3)
                image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (0, 255, 0), 2)
                continue

            image = cv2.rectangle(image, tuple(bbox[:2][::-1]), tuple(bbox[2:][::-1]), (255, 0, 0), 2)

            #image = cv2.rectangle()


    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite('test_448_single_process.jpg', image)


def draw(img_fp, coord):
    image = cv2.imread(img_fp, -1)
    coords = np.load(coord)
    print(coords.shape)
    for coord in coords:
        #print(coord)
        image = cv2.circle(image, tuple(coord[::-1] / 2), 3, (0, 200, 0), cv2.FILLED, 1)

    cv2.imwrite('test1.jpg', image)


def json2image(image_folder, json_path):
    json_basename = os.path.basename(json_path)
    json_prefix = json_basename.split('.')[0]
    for i in glob.iglob(os.path.join(image_folder, '**', '*.tif'), recursive=True):
        if json_prefix in i:
            return i




if __name__ == '__main__':
    #p = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/raw/CRC/fold_2/2_low_grade/H06-04442_A5H_E_1_1_grade_2_0673_0225.png'
    #c = '/data/smb/syh/PycharmProjects/CGC-Net/data_yanning/proto/coordinate/CRC/fold_2/2_low_grade/H06-04442_A5H_E_1_1_grade_2_0673_0225.npy'
    #p = '/data/by/tmp/HGIN/test_can_be_del3/fold_1/1_normal/Grade1_Patient_157_5_grade_1_row_4256_col_0896.png'
    #p = '/home/baiyu/test_can_be_del3/fold_1/1_normal/Grade1_Patient_135_097357_020131_grade_1_row_0000_col_3360.png'
    #p = '/home/baiyu/test_can_be_del3/'
    #p = '/home/baiyu/tmp/hover_net/prostate_images'
    #p = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug/ZT111_4_C_5_11_crop_2_grade_1_aug_6.jpg'

    #p = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops_Aug/'
    #p = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/'
    #p = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/5Crops'

    #c = '/data/by/tmp/hover_net/samples/out/backup_json/Grade3_Patient_039_080185_016871_grade_3_row_3584_col_6720.json'
    #c = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/fold_1/1_normal/json/Grade1_Patient_157_5_grade_1_row_4256_col_0896.json'
    #c = '/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/json/'
    #c = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops'
    #c = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/5Crops_Aug'
    p = '/data/smb/syh/tmp/'
    c = '/data/smb/syh/tmp/'
    p = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images_Aug/'
    c = '/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json_Aug/'


    jsons = glob.glob(os.path.join(c, '**', '*.json'), recursive=True)
    #print(len(jsons))
    json_fp = random.choice(jsons)
    #json_fp = jsons[1]
    #print(json_fp)
    image_fp = json2image(p, json_fp)
    vis(image_fp, json_fp)
    print(image_fp)
    #print(image_fp)

    #import random
    #c = '/data/by/tmp/hover_net/samples/out/json/Grade3_Patient_039_080185_016871_grade_3_row_3584_col_6720.json'
    #vis(image_fp, json_fp)
    #vis('/data/by/tmp/hover_net/samples/input/H06-04442_A5H_E_1_1_grade_2_0673_0225.png', '/data/by/tmp/hover_net/samples/out/json/H06-04442_A5H_E_1_1_grade_2_0673_0225.json')

    import sys;sys.exit()

    from stich import BaseDataset, ImageFolder
    image_folder = ImageFolder('/data/by/tmp/HGIN/test_can_be_del2')
    dataset = BaseDataset(image_folder, 1792)
    #file_path_list1 = dataset.image_path_lists
    #print(len(file_path_list1))

    file_path_list = []
    count = 0
    crc_extends_images = []
    #for image, path in dataset:
    path = '/data/by/tmp/hover_net/samples/out/json/Grade3_Patient_172_7_grade_3_row_0224_col_1120.json'
    #path = 'Grade3_Patient_172_7_grade_3_row_0000_col_5152.json'
    image_name = os.path.basename(path).split('.')[0] + '.png'
    #print(len(dataset))
    #path = '/data/by/tmp/HGIN/test_can_be_del2/fold_3/3_high_grade/Grade3_Patient_172_7_grade_3_row_0000_col_0000.png
    image_path = dataset.image_path_lists
    print(len(image_path))
    #print(os.path.dirname(image_path[0]))
    #print(image_path[0])
    image_path = os.path.join(os.path.dirname(image_path[0]), image_name)
    #print(image_path)
    _, image = dataset.get_image_by_path(image_path)
    print(image.shape)
    #print(path.shape)
    #print(image)
    vis(image, path)
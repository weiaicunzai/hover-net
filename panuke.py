import numpy as np
import cv2
import glob
import os
from pathlib import Path



#img_fp = '/data/by/datasets/original/PanNuke/Fold 1/images/fold1/images.npy'
#mask_fp = '/data/by/datasets/original/PanNuke/Fold 1/masks/fold1/masks.npy'


#img = np.load(img_fp)
#mask = np.load(mask_fp)
#
#print(img.shape)
#print(mask.shape)
#
#image = img[30]
#
#mask = mask[30]

def write_masks(image, mask):
    for i in range(6):
        m = mask[:, :, i]
        if m.max() != 0:
            m = m / m.max() * 255

        cv2.imwrite('{}.jpg'.format(i), m)

    cv2.imwrite('{}.jpg'.format('aa'), image)


def convert_label(mask):
    """mask: shape 256 256 6"""
    chl = mask.shape[-1] - 1 # no background

    output = np.zeros(mask.shape[:2])

    count = 0
    for c in range(chl):
        uni = np.unique(mask[:, :, c])
        for ins in uni:
            if ins == 0:
                continue
            count += 1
            output[mask[:, :, c] == ins] = count

    return output

if __name__ == '__main__':
    panuke_path = '/data/by/datasets/original/PanNuke'
    save_path = '/data/smb/syh/colon_dataset/Panuke/'
    panuke_path = Path(panuke_path)
    save_path = Path(save_path)

    for i in glob.iglob(os.path.join(panuke_path, '**', 'images.npy'), recursive=True):
        #print(i)
        #print(i)
        #print(panuke_path.parts)
        images_path = Path(i)
        masks_path = Path(i.replace('images', 'masks'))
        images = np.load(images_path)
        masks = np.load(masks_path)
        #images_path_len = len(images_path.parts)
        #print(images_path.parts)
        #a = images_path.parts
        #print(os.path.join(*a))
        prefix = panuke_path.parts
        prefix_len = len(prefix)
        images_sub = images_path.parts[prefix_len:-1]
        masks_sub = masks_path.parts[prefix_len:-1]
        #print(os.path.join(*masks_sub))
        #print(panuke_path)
        #print(i)
        images_save_prefix = Path(os.path.join(save_path, *images_sub))
        #images_save_prefix = Path(images_save_prefix)
        masks_save_prefix = Path(os.path.join(save_path, *masks_sub))
        #masks_save_prefix = Path(masks_save_prefix)
        #print(images_save_prefix)
        #print(masks_save_prefix)
        #mask_save_prefix = os.path.join()

        #image_save_path = os.path.join(save_path, *)

        #import sys; sys.exit()
        images_save_prefix.mkdir(parents=True, exist_ok=True)
        masks_save_prefix.mkdir(parents=True, exist_ok=True)


        for idx, (image, mask) in enumerate(zip(images, masks)):
            #print('before:', np.unique(mask[:, :, 0:-1]))
            #write_masks(image, mask)
            mask = convert_label(mask)

            image_name = 'img_{}.jpg'.format(idx)
            image_save_path = os.path.join(images_save_prefix, image_name)
            print(image_save_path)
            cv2.imwrite(image_save_path, image)
            mask_name = 'mask_{}.png'.format(idx)
            mask_save_path = os.path.join(masks_save_prefix, mask_name)
            print(mask_save_path)
            cv2.imwrite(mask_save_path, mask)
            #print(np.unique(mask))

            # write
            #mask = mask / mask.max() * 255
            #cv2.imwrite('after.jpg', mask)

            #if idx == 133:
                #import sys; sys.exit()

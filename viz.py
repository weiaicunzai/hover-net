import cv2
import numpy as np
import random




#np_file = '/data/by/datasets/original/MoNuSAC/MoNuSAC_images_and_annotations/TCGA-J4-A67T-01Z-00-DX1/TCGA-J4-A67T-01Z-00-DX1-3.npy'
#img_file = '/data/by/datasets/original/MoNuSAC/MoNuSAC_images_and_annotations/TCGA-J4-A67T-01Z-00-DX1/TCGA-J4-A67T-01Z-00-DX1-3.tif'
#
#
#img = cv2.imread(img_file, -1)
#
#mask = np.load(np_file)
##print(np.unique(mask))
#print(img.shape, mask.shape)
#
#img[mask != 0] = 255
#
#cv2.imwrite('aa.jpg', img)

#image = cv2.imread('/data/smb/syh/colon_dataset/Panuke/Fold 1/images/fold1/img_598.jpg', -1)
#mask = cv2.imread('/data/smb/syh/colon_dataset/Panuke/Fold 1/masks/fold1/mask_598.png', -1)
#
#print(np.unique(mask))
#
#mask = mask / mask.max() * 255
#
#cv2.imwrite('0.jpg', image)
#cv2.imwrite('1.png', mask)
from extract_patches_all import MoNuSAC, Kumar, CPM17, TNBC, combined


#dataset = MoNuSAC('/data/smb/syh/colon_dataset/MoNuSAC/MoNuSAC_images_and_annotations/')
#dataset =  CPM17('/data/smb/syh/colon_dataset/CPM17')
#dataset = TNBC('/data/smb/syh/colon_dataset/TNBC')
#print(len(dataset))


#image, mask = dataset[33]

def overlay(image, mask):
    mask[mask != 0] = 255
    kernel = np.ones((3,3), np.uint8)
    mask_dilation = cv2.dilate(mask.astype('uint8'), kernel, iterations=1)
    mask = cv2.bitwise_xor(mask_dilation, mask.astype('uint8'))
    print(image.shape)
    image[mask != 0] = (0, 0, 255)
    return image
    #cv2.imwrite('0.jpg', image)

def viz(image, mask):
    print(np.unique(mask))
    mask = mask / mask.max() * 255
    mask = cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2BGR)
    #if image.shape[-1] == 4:
    #    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    print(image.shape, mask.shape)
    return cv2.hconcat([image.astype('uint8'), mask.astype('uint8')])


def compress_nine(images):
    imgs = []
    for image in images:
        imgs.append(cv2.resize(image, (256, 256)))

    img1 = cv2.hconcat(imgs[:3])
    img2 = cv2.hconcat(imgs[3:6])
    img3 = cv2.hconcat(imgs[6:])
    img = cv2.vconcat([img1, img2, img3])
    return img
#output = viz(*dataset[44])
if __name__ == '__main__':
    combined_train, combined_test = combined.split_train_valid([0.9, 0.1])
    print(len(combined_train))
    print(len(combined_test))
    dataset = combined_train
    res = random.sample(range(len(dataset)), k=9)
    print(res)

    imgs = []
    for idx in res:
        imgs.append(overlay(*dataset[idx]))

    img = compress_nine(imgs)
    cv2.imwrite('0.jpg', img)



    #for idx, i in enumerate(dataset):
        #output = overlay(*i)
        #cv2.imwrite('aa/{}.jpg'.format(idx), output)
#cv2.imwrite('0.jpg', image)
#cv2.imwrite('1.png', mask)
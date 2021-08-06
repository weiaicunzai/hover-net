"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
from scipy import io

from pathlib import Path
import numpy as np
import cv2
import torch

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

#dataset_info = {
#    'CoNsep':
#}
class PanNuke:
    def __init__(self, path):
        self.images = []
        self.masks = []
        path = Path(path)
        for i in glob.iglob(os.path.join(path, '**', '*.jpg'), recursive=True):
            image_path = Path(i)

            sub_path = str(image_path.relative_to(path))
            sub_path = sub_path.replace('images', 'masks')
            sub_path = sub_path.replace('img', 'mask')
            sub_path = sub_path.replace('jpg', 'png')
            #mask_path = Path(os.path.join(path, sub_path))
            self.images.append(str(image_path))
            self.masks.append(os.path.join(path, sub_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        mask = cv2.imread(mask_path, -1)
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")

        return image, mask


class CoNSeP:
    def __init__(self, path):
        self.images = []
        self.masks = []
        path = Path(path)
        for i in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            image_path = Path(i)

            sub_path = str(image_path.relative_to(path))
            sub_path = sub_path.replace('Images', 'Labels')
            #sub_path = sub_path.replace('img', 'mask')
            sub_path = sub_path.replace('.png', '.mat')
            if 'Overlay' in sub_path:
                continue
            #print(sub_path)
            #mask_path = Path(os.path.join(path, sub_path))
            self.images.append(str(image_path))
            self.masks.append(os.path.join(path, sub_path))
            #print(self.images[-1], self.masks[-1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        #mask = io.loadmat()
        mask = io.loadmat(mask_path)["inst_map"] #[1000, 1000]
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")
        #print(mask.shape, image.shape)

        return image, mask

class MoNuSAC:
    def __init__(self, path):
        self.images = []
        self.masks = []
        for i in glob.iglob(os.path.join(path, '**', '*.tif')):
            self.images.append(i)
            mask = i.replace('.tif', '.npy')
            self.masks.append(mask)

            #assert os.path.exists(mask), True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        #import sys; sys.exit()
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        #mask = io.loadmat()
        mask = np.load(mask_path) #[1000, 1000]
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")
        #print(mask.shape, image.shape)

        return image, mask

class Kumar:
    def __init__(self, path):
        self.images = []
        self.masks = []
        path = Path(path)
        for i in glob.iglob(os.path.join(path, '**', '*.tif'), recursive=True):
            image_path = Path(i)

            sub_path = str(image_path.relative_to(path))
            sub_path = sub_path.replace('Images', 'Labels')
            #sub_path = sub_path.replace('img', 'mask')
            sub_path = sub_path.replace('.tif', '.mat')
            if 'Overlay' in sub_path:
                continue
            #print(sub_path)
            #mask_path = Path(os.path.join(path, sub_path))
            self.images.append(str(image_path))
            self.masks.append(os.path.join(path, sub_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        #mask = io.loadmat()
        mask = io.loadmat(mask_path)["inst_map"] #[1000, 1000]
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")
        #print(mask.shape, image.shape)

        return image, mask

class CPM17:
    def __init__(self, path):
        self.images = []
        self.masks = []
        path = Path(path)
        for i in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            image_path = Path(i)

            sub_path = str(image_path.relative_to(path))
            sub_path = sub_path.replace('Images', 'Labels')
            #sub_path = sub_path.replace('img', 'mask')
            sub_path = sub_path.replace('.png', '.mat')
            if 'Overlay' in sub_path:
                continue
            #print(sub_path)
            #mask_path = Path(os.path.join(path, sub_path))
            self.images.append(str(image_path))
            self.masks.append(os.path.join(path, sub_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        #mask = io.loadmat()
        mask = io.loadmat(mask_path)["inst_map"] #[1000, 1000]
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")
        #print(mask.shape, image.shape)

        return image, mask

class TNBC:
    def __init__(self, path):
        self.images = []
        self.masks = []
        path = Path(path)
        for i in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
            image_path = Path(i)

            sub_path = str(image_path.relative_to(path))
            if '5784' not in sub_path:
                continue
            sub_path = sub_path.replace('Images', 'Labels')
            #sub_path = sub_path.replace('img', 'mask')
            sub_path = sub_path.replace('.png', '.mat')
            sub_path = list(Path(sub_path).parts)
            sub_path.remove('5784')
            sub_path = os.path.join(*sub_path)
            if 'Overlay' in sub_path:
                continue
            #print(sub_path)
            #mask_path = Path(os.path.join(path, sub_path))
            self.images.append(str(image_path))
            self.masks.append(os.path.join(path, sub_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        print(image_path, mask_path)
        image = cv2.imread(image_path, -1)
        #mask = io.loadmat()
        mask = io.loadmat(mask_path)["inst_map"] #[1000, 1000]
        mask = np.expand_dims(mask, -1)
        mask = mask.astype("int32")
        #print(mask.shape, image.shape)

        return image, mask
config = {
    '/data/smb/syh/colon_dataset/Panuke': PanNuke,
    '/data/smb/syh/colon_dataset/CoNSeP': CoNSeP,
    '/data/smb/syh/colon_dataset/CPM17': CPM17,
    '/data/smb/syh/colon_dataset/Kumar': Kumar,
    '/data/smb/syh/colon_dataset/MoNuSAC': MoNuSAC,
    '/data/smb/syh/colon_dataset/TNBC': TNBC,
}

class ComBinedFactory:
    def __init__(self, config):
        datasets = []
        for k, v in config.items():
            datasets.append(v(k))

        print(datasets)
        from torch.utils.data.dataset import ConcatDataset
        self.dataset = ConcatDataset(datasets)
        #print(len(dataset))
    def split_train_valid(self, split_ratio):
        total = len(self.dataset)
        r1 = int(total * split_ratio[0])
        r2 = total - r1

        if sum(split_ratio) != 1:
            raise ValueError('split_ratio should be summed to one')

        return torch.utils.data.random_split(
            self.dataset,
            [r1, r2],
            generator=torch.Generator().manual_seed(42))


combined = ComBinedFactory(config)
#print(len(combined_train))
#print(len(combined_test))





#class ComBined:
#    def __init__(self)
#class ConcatDataset(Dataset[T_co]):
#    r"""Dataset as a concatenation of multiple datasets.
#
#    This class is useful to assemble different existing datasets.
#
#    Args:
#        datasets (sequence): List of datasets to be concatenated
#    """
#    datasets: List[Dataset[T_co]]
#    cumulative_sizes: List[int]
#
#    @staticmethod
#    def cumsum(sequence):
#        r, s = [], 0
#        for e in sequence:
#            l = len(e)
#            r.append(l + s)
#            s += l
#        return r
#
#    def __init__(self, datasets: Iterable[Dataset]) -> None:
#        super(ConcatDataset, self).__init__()
#        # Cannot verify that datasets is Sized
#        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
#        self.datasets = list(datasets)
#        for d in self.datasets:
#            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
#        self.cumulative_sizes = self.cumsum(self.datasets)
#
#    def __len__(self):
#        return self.cumulative_sizes[-1]
#
#    def __getitem__(self, idx):
#        if idx < 0:
#            if -idx > len(self):
#                raise ValueError("absolute value of index should not exceed dataset length")
#            idx = len(self) + idx
#        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#        if dataset_idx == 0:
#            sample_idx = idx
#        else:
#            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
#        return self.datasets[dataset_idx][sample_idx]
#
#    @property
#    def cummulative_sizes(self):
#        warnings.warn("cummulative_sizes attribute is renamed to "
#                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
#        return self.cumulative_sizes

#'/data/smb/syh/colon_dataset/Panuke'
#'/data/smb/syh/colon_dataset/CoNSeP/'
#'/data/smb/syh/colon_dataset/MoNuSAC/'
#pancake = PanNuke('/data/smb/syh/colon_dataset/Panuke')
#consep = CoNSeP('/data/smb/syh/colon_dataset/CoNSeP/')
#dataset = MoNuSAC('/data/smb/syh/colon_dataset/MoNuSAC/MoNuSAC_images_and_annotations/')
#dataset = MoNuSAC('/data/smb/syh/colon_dataset/MoNuSAC/MoNuSAC_images_and_annotations/')
#print(dataset[33])
#pancake[330]
#dataset[33]
# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    #type_classification = True
    type_classification = False

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "combined"
    #save_root = "dataset/training_data/%s/" % dataset_name
    save_root = "/data/smb/syh/colon_dataset/hovernet_training_data/%s/" % dataset_name
    combined_train, combined_test = combined.split_train_valid([0.9, 0.1])

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        #"train": {
            #"img": (".png", "dataset/CoNSeP/Train/Images/"),
            #"ann": (".mat", "dataset/CoNSeP/Train/Labels/"),
            #"img": (".png", "/data/by/datasets/original/CoNSep/CoNSeP/Train/Images/"),
            #"ann": (".mat", "/data/by/datasets/original/CoNSep/CoNSeP/Train/Labels/"),
        #},
        #"valid": {
        #    #"img": (".png", "dataset/CoNSeP/Test/Images/"),
        #    #"ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
        #    "img": (".png", "/data/by/datasets/original/CoNSep/CoNSeP/Test/Images/"),
        #    "ann": (".mat", "/data/by/datasets/original/CoNSep/CoNSeP/Test/Labels/"),
        #},
        "train" : combined_train,
        "valid" : combined_test
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    #parser = get_dataset(dataset_name)
    # print(win_size, step_size, 1111) [540, 540] [164, 164] 1111
    xtractor = PatchExtractor(win_size, step_size)
    #for split_name, split_desc in dataset_info.items():
    for split_name, dataset in dataset_info.items():
        #img_ext, img_dir = split_desc["img"]
        #ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        #file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        #file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(dataset), bar_format=pbar_format, ascii=True, position=0
        )

        #for file_idx, file_path in enumerate(file_list):
        for file_idx, (img, ann) in enumerate(dataset):
            #base_name = pathlib.Path(file_path).stem
            base_name = "{}_{}".format(split_name, file_idx)
            #print(file_path, base_name)

            #img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            #ann = parser.load_ann(
            #    "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            #)

            # *
            #print(ann.shape)
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()

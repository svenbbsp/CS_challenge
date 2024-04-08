from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
import os
import glob

from GTA5.labels import GTA5Labels_TaskCV2017


class GTA_5(torchDataset):
    label_map = GTA5Labels_TaskCV2017()

    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root):
            self.root = root
            self.img_paths = self.create_imgpath_list()
            self.lbl_paths = self.create_lblpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            img_dir = os.path.join(self.root , self.IMG_DIR_NAME)
            file_paths = []
            for root, _, files in os.walk(img_dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))
            
            return file_paths

        def create_lblpath_list(self):
            lbl_dir =  os.path.join(self.root , self.LBL_DIR_NAME)
            file_paths = []
            for root, _, files in os.walk(lbl_dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))
            
            return file_paths

    def __init__(self, root: Path, transform):
        """

        :param root: (Path)
            this is the directory path for GTA5 data
            must be the following
            e.g.)
                ./data
                ├── images
                │  ├── 00001.png
                │  ├── ...
                │  └── 24966.png
                ├── images.txt
                ├── labels
                │  ├── 00001.png
                │  ├── ...
                │  └── 24966.png
                ├── test.txt
                └── train.txt
        """
        self.root = root
        self.transform = transform
        self.paths = self.PathPair_ImgAndLabel(root=self.root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path

        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)

        img = self.transform(img)
        lbl = self.transform(lbl)*255
        return img, lbl

    @staticmethod
    def read_img(path):
        img = Image.open(str(path))
        img = np.array(img)
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        # remap_lbl = lbl[np.where(np.isin(lbl, cls.label_map.support_id_list), lbl, 0)]
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl
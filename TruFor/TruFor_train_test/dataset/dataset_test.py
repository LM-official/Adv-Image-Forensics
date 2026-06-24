# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, list_img=None, max_pixels=0):
        self.img_list = list_img
        # downscale images above this pixel count (0 disables); full-resolution
        # multi-megapixel images can exceed the memory of smaller machines
        self.max_pixels = max_pixels

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        assert self.img_list
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_path = self.img_list[index]
        img = Image.open(rgb_path).convert("RGB")
        orig_size = (img.height, img.width)
        if self.max_pixels and img.width * img.height > self.max_pixels:
            scale = (self.max_pixels / (img.width * img.height)) ** 0.5
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        img_RGB = np.array(img)
        return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path, torch.tensor(orig_size)

    def get_filename(self, index):
        item = self.img_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item
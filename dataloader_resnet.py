import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from skimage import io
import random

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs_dir = imgs_dir
        self.transformations = transformations
        self.imgs_ids = [file for file in listdir(imgs_dir)]
        random.shuffle(self.imgs_ids) # shuflle the list

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        img_file = self.imgs_dir + img_idx
        if "virals" in img_file:
            label = torch.tensor(1)
        elif "covid" in img_file:
            label = torch.tensor(2)
        else:
            label = torch.tensor(0)
        img = io.imread(img_file)
        if self.transformations:
            img = self.transformations(img)

        # Standarize
        if len(img.shape) == 2:
            img = (img - torch.mean(img)) / torch.std(img)
        return {"image": img, "label": label}

    def __len__(self):
        return len(self.imgs_ids)


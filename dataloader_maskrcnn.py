import numpy as np
import torch
from torch.utils.data import Dataset
from os import listdir
from numpy import clip
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, transformations=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transformations = transformations
        self.imgs_ids = [file for file in listdir(imgs_dir)]
        self.mask_ids = [file for file in listdir(masks_dir)]

    @classmethod
    def preprocess(cls, img, expand_dim=False, adjust_label=False, normalize=False):
        # Normalize
        if normalize:
            # Normalize
            if img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min())
            # Global standarize
            if len(img.shape) == 2:
                img = (img - img.mean()) / img.std()
                img = clip(img, -1.0, 1.0)
                img = (img + 1.0) / 2.0
            else:
                # Channel-wise standarization
                means = img.mean(axis=(0, 1), dtype='float64')
                stds = img.std(axis=(0, 1), dtype='float64')
                img = (img - means) / stds
                img = clip(img, -1.0, 1.0)
                img = (img + 1.0) / 2.0

        # Expand dimensions for image if specified
        if expand_dim:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
            img = img.transpose((2, 0, 1))

        if adjust_label:
            coords = np.where(img != 0)
            img[coords] = 1
        return img

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        img_file = self.imgs_dir + img_idx
        mask_file = self.masks_dir + mask_idx
        # Read image and mask
        img = Image.open(img_file).convert("RGB")
        img = asarray(img)
        mask = np.load(mask_file)
        if self.transformations:
            data = self.transformations({"img": img, "mask": mask})
            img, mask = data['img'], data['mask']

        # Get the preprocessed images and apply transformations if specified
        img = self.preprocess(img, expand_dim=False, adjust_label=False, normalize=True)
        # Compute bounding box
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for index in range(num_objs):
            pos = np.where(masks[index])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # e.g. tensor([[113., 153., 439., 382.]])
        labels = torch.ones((num_objs,), dtype=torch.int64)  # e.g. tensor([1])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img_id = torch.tensor([i])  # tensor([1])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Set up annotations
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["masks"] = masks
        my_annotation["area"] = area
        my_annotation["iscrowd"] = iscrowd
        return torch.from_numpy(img), my_annotation

    def __len__(self):
        return len(self.imgs_ids)




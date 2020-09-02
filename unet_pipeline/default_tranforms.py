import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise, img_as_ubyte
import random
import cv2
import torch
from skimage.measure import label
from skimage import measure

def make_binary(mask):
    coords = np.where(mask != 0)
    mask[coords] = 1
    return mask

def assign_labels(mask):
    coords = np.where(mask != 0)
    mask[coords] = 1
    labeled = label(mask)
    regions = measure.regionprops(labeled)
    if len(regions) > 2:
        areas = [r.area for r in measure.regionprops(labeled)]
        areas.sort()
        for region in measure.regionprops(labeled):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    labeled[coordinates[0], coordinates[1]] = 0
        coords = np.where(labeled != 0)
        labeled[coords] = 1
        new_labeled = label(labeled)
        return new_labeled
    else:
        return labeled

class RandomRotate:
    def __call__(self, data):
        rotation_angle = random.randint(-180, 180)
        img, mask = data['img'], data['mask']
        img = rotate(img, rotation_angle, mode='reflect').astype(float)
        mask = assign_labels(rotate(img_as_ubyte(mask), rotation_angle, mode='reflect')).astype(float)
        return {"img": img, "mask": mask}

class HorizontalFlip:
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        h_img = np.fliplr(img).astype(float)
        h_mask = np.fliplr(mask).astype(float)
        return {"img": h_img, "mask": h_mask}

class VerticalFlip:
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        v_img = np.flipud(img).astype(float)
        v_mask = np.flipud(mask).astype(float)
        return {"img": v_img, "mask": v_mask}

class RandomNoise:
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        noised_img = random_noise(img).astype(float)
        mask = mask.astype(float)
        return {"img": noised_img, "mask": mask}

class RandomBlur(object):
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        blur_factor = random.randrange(1, 10, 2)
        blurred_img = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
        return {"img": blurred_img, "mask": mask}

class ToTensor(object):
    def __call__(self, data):
        img, mask = data['img'], data['mask']
        tensored = torch.from_numpy(img)
        return {"img": tensored, "mask": mask}


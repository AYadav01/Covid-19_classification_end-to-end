from torch.utils.data.dataset import Dataset
from PIL import Image
from os import listdir

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, transformation=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transformations = transformation
        self.imgs_ids = [file for file in listdir(imgs_dir)]
        self.mask_ids = [file for file in listdir(masks_dir)]

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        image_path = self.imgs_dir + img_idx
        mask_path = self.masks_dir + mask_idx
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        if self.transformations is not None:
            image = self.transformations(image)
            mask = self.transformations(mask)
        return image, mask

    def __len__(self):
        return len(self.imgs_ids)



import numpy as np
import os
import json
from PIL import Image, ImageDraw

def _read_coords(image_size, lung_1, lung_2):
    img_1, img_2 = Image.new('L', image_size, 0), Image.new('L', image_size, 0)
    polygon_1 = []
    polygon_2 = []
    # Append coordinates of both lungs
    for arg in lung_1:
        polygon_1.append((arg['x'], arg['y']))
    for arg in lung_2:
        polygon_2.append((arg['x'], arg['y']))
    # Draw polygons
    ImageDraw.Draw(img_1).polygon(polygon_1, outline=1, fill=1)
    ImageDraw.Draw(img_2).polygon(polygon_2, outline=2, fill=2)
    # Convert to numpy
    mask_1 = np.array(img_1)
    mask_2 = np.array(img_2)
    final_mask = mask_1 + mask_2
    return final_mask

def reconstruct_mask(annotations, save_path):
    files = os.listdir(annotations)
    for index, arg in enumerate(files):
        if index <= len(files):
            full_path = os.path.join(annotations, arg)
            with open(full_path) as f:
                data = json.load(f)
                file_name = arg.split(".json")[0]+".npy"
                image_size = (data["image"]["width"], data["image"]["height"])
                try:
                    # 0 & 1 have coords for lungs and rest are other tags
                    lung_1 = data["annotations"][0]["polygon"]["path"]
                    lung_2 = data["annotations"][1]["polygon"]["path"]
                    mask = _read_coords(image_size, lung_1, lung_2)
                    save_mask_path = os.path.join(save_path, file_name)
                    np.save(save_mask_path, mask)
                except Exception as e:
                    print("Error with annotation: {} - {}".format(arg, e))
    print("Mask Created!")


if __name__ == "__main__":
    covid_annotations = "data\\covid\\annotations"
    normal_annotations = "data\\normal\\annotations"
    viral_annotations = "C:\\Users\\AnilYadav\\Desktop\\virals\\annotations"
    save_path = "C:\\Users\\AnilYadav\\Desktop\\virals\\npy_masks"
    reconstruct_mask(viral_annotations, save_path)


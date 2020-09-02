import SimpleITK as sitk
import six
import matplotlib.pyplot as plt
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, gldm
from skimage import io
import csv
import numpy as np
import os
from skimage.measure import label, regionprops
from tqdm import tqdm
from PIL import Image
from numpy import asarray
from skimage.color import rgb2gray
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

def tqdm_enumerate(iterator):
    i = 0
    for y in tqdm(iterator):
        yield i, y
        i += 1

def get_extractor(img, mask, settings, name=None):
    if name:
        features = {"first_order": firstorder.RadiomicsFirstOrder(img, mask, **settings),
                    "glcm": glcm.RadiomicsGLCM(img, mask, **settings),
                    "glrm": glrlm.RadiomicsGLRLM(img, mask, **settings),
                    "glszm": glszm.RadiomicsGLSZM(img, mask, **settings),
                    "gldm": gldm.RadiomicsGLDM(img, mask, **settings)
                    }
        return features[name]

def extract_features(img, mask, classes, settings, fileName, writeHeader=False):
    # Crop the image to correspond to the mask
    try:
        # bb, correctedMask = imageoperations.checkMask(img, mask, label=1)
        # if correctedMask is not None:
        #     mask = correctedMask
        # croppedImage, croppedMask = imageoperations.cropToTumorMask(img, mask, bb)
        croppedImage, croppedMask = img, mask
        header, values = [], []
        if writeHeader:
            header.append("slice_id")
        values.append(str(fileName))
        for index, arg in enumerate(classes):
            feature = get_extractor(croppedImage, croppedMask, settings, arg)
            feature.enableAllFeatures()
            result = feature.execute()
            # Writing to File
            for (key, val) in six.iteritems(result):
                if writeHeader:
                    header.append(str(key))
                values.append(val.item())

        if writeHeader:
            header.append('label')
            # Class Label
        values.append(1)
    except Exception as e:
        header, values = None, None
        print("File {} skipped due to {}".format(fileName, str(e)))
    finally:
        if writeHeader:
            return header, values
        else:
            return values

def process_files(img_path=None, mask_path=None):
    if img_path and mask_path:
        # Define settings and class of features to extract
        setting = {}
        setting['binWidth'] = 25
        setting['label'] = 1
        setting['interpolator'] = 'sitkBSpline'
        setting['resampledPixelSpacing'] = None
        setting['weightingNorm'] = None
        classes = ["first_order", "glcm"]

        with open("radiomic_features.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for index, arg in tqdm_enumerate(img_path):
                if index <= len(img_path):
                    fileName = arg.split("\\")[-1]
                    # Load image and mask
                    img = rgb2gray(io.imread(arg))
                    mask = np.load(mask_path[index])
                    coords = np.where(mask != 0)
                    mask[coords] = 1

                    # Check we have two distinct regions
                    label_image = label(mask)
                    propsa = regionprops(label_image)
                    if len(propsa) == 2:
                        img = sitk.GetImageFromArray(img)
                        mask = sitk.GetImageFromArray(mask)
                        # Extract features
                        if index == 0:
                            writeHeader = True
                            header, values = extract_features(img, mask, classes, setting, fileName,
                                                              writeHeader)
                            if values is not None:
                                writer.writerow(header)
                                writer.writerow(values)
                        else:
                            writeHeader = False
                            values = extract_features(img, mask, classes, setting, fileName, writeHeader)
                            if values is not None:
                                writer.writerow(values)
                    else:
                        print("Error for file:", fileName)


if __name__ == "__main__":
    image_path = "path_to_images"
    mask_path = "path_to_masks"
    images = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    masks = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
    process_files(images, masks)


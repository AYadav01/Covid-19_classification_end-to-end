import numpy as np
import os
from skimage import io

left_path = "data\\montgomery\\ManualMask\\leftMask"
left = os.listdir(left_path)
right_path = "data\\montgomery\\ManualMask\\rightMask"
right = os.listdir(right_path)
save_path = "data\\montgomery\\combined"

for index in range(len(left)):
    # Read masks
    left_lung = io.imread(os.path.join(left_path, left[index]))
    right_lung = io.imread(os.path.join(right_path, right[index]))

    # Build save path
    file_name = left[index].split("\\")[-1].split(".")[0] + ".npy"
    file_path = os.path.join(save_path, file_name)

    # Lable left lung as 1 and right lung as 2
    left_coord = np.where(left_lung != 0)
    left_lung[left_coord] = 1
    right_coord = np.where(right_lung != 0)
    right_lung[right_coord] = 2

    # Combine masks
    mask = left_lung + right_lung
    np.save(file_path, mask)

print("ALL MASKS COMBINED!")



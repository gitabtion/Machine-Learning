import numpy as np
import cv2
import os
from PIL import Image

patch_height = 28
patch_width = 28
patch_size = patch_height * patch_width

boat_patches_path = "../data/train/patches/boat/"
water_patches_path = "../data/train/patches/water/"
boat_img_path = "../data/train/boat/"
water_img_path = "../data/train/water/"
boat_img_files = os.listdir(boat_img_path)
water_img_files = os.listdir(water_img_path)

for file in boat_img_files:
    if not os.path.isdir(file):
        img = Image.open(boat_img_path + file)
        for x in range(0, img.size[0] - patch_width, 10):
            for y in range(0, img.size[1] - patch_height, 10):
                img.crop((x, y, x + patch_width, y + patch_height)).save(boat_patches_path + str(x) + str(y) + file)

for file in water_img_files:
    if not os.path.isdir(file):
        img = Image.open(water_img_path + file)
        for x in range(0, img.size[0] - patch_width, 10):
            for y in range(0, img.size[1] - patch_height, 10):
                img.crop((x, y, x + patch_width, y + patch_height)).save(water_patches_path + str(x) + str(y) + file)

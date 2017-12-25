import os
import random

import shutil

boat_train_patches_path = "../../../mcm_prime/data/train/patches/boat/"
water_train_patches_path = "../../../mcm_prime/data/train/patches/water/"
boat_test_patches_path = "../../../mcm_prime/data/test/boat/"
water_test_patches_path = "../../../mcm_prime/data/test/water/"
boat_train_files = os.listdir(boat_train_patches_path)
boat_test_files = os.listdir(boat_test_patches_path)
water_train_files = os.listdir(water_train_patches_path)
water_test_files = os.listdir(water_test_patches_path)
b_len = 50000
w_len = 20000

for i in range(0, 5000):

    b = random.randint(0, b_len)
    if not os.path.exists(boat_test_patches_path + boat_train_files[b]):
        shutil.move(boat_train_patches_path + boat_train_files[b], boat_test_patches_path)
        b_len -= 1
    w = random.randint(0, w_len)
    if not os.path.exists(water_test_patches_path + water_train_files[w]):
        shutil.move(water_train_patches_path + water_train_files[w], water_test_patches_path)
        w_len -= 1

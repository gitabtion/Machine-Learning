import numpy as np
import cv2
from PIL import Image

# tag R0 G255 B30


img = cv2.imread("./boder.jpg", 3)
result_img = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
img[10][10] = [0, 255, 30]
for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        if (img[x][y] == [0, 255, 30]).all():
            result_img[x][y] = [255, 255, 255]
        if x > 0 and (result_img[x - 1][y] == [255, 255, 255]).all():
            result_img[x][y] = [255, 255, 255]

cv2.imwrite("./result2.jpg", result_img)

import cv2
from PIL import Image
import numpy as np

img = cv2.imread("./test.jpg", 3)
img2 = img
img = cv2.GaussianBlur(img, (3, 3), 3)
print(img.shape)
for x in range(0, img.shape[0]):
    for y in range(0, img.shape[1]):
        if (img[x][y] != [255, 255, 255]).any():
            img[x][y] = [0, 0, 0]
# img[(img[:, :] < [255, 255, 255]).any(), :] = [0, 0, 0]
# img = cv2.Canny(img, 50, 150)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2、固定阈值二值化
retval, im_at_fixed = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
# 将阈值设置为50，阈值类型为cv2.THRESH_BINARY，则灰度在大于50的像素其值将设置为255，其它像素设置为0


print(img.shape)
print("done")
cv2.imwrite("./gray.jpg", img)
cv2.imshow("01", im_at_fixed)
print("done")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# im_at_mean = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
cv2.imwrite("./sd.jpg", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



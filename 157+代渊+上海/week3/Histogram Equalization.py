#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ֱ��ͼ���⻯
def equalizeHist(img):
    height, width = img.shape
    emptyImage = np.zeros((height, width), np.uint8)
    sumPi = 0
    for i in range(255):
        Ni = 0
        for src_x in range(width):
            for src_y in range(height):
                if img[src_x,src_y] == i:
                    Ni = Ni + 1
        Pi = Ni/(height*width)
        sumPi = sumPi + Pi
        q = sumPi*256 - 1

        for dst_x in range(width):
            for dst_y in range(height):
                if img[dst_x, dst_y] == i:
                    emptyImage[dst_x,dst_y] = q
    return emptyImage


# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = equalizeHist(gray)

# ֱ��ͼ
#hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


'''
# ��ɫͼ��ֱ��ͼ���⻯
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
(b, g, r) = cv2.split(img)
bH = equalizeHist(b)
gH = equalizeHist(g)
rH = equalizeHist(r)
# �ϲ�ÿһ��ͨ��
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''

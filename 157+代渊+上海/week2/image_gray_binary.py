# -*- coding: utf-8 -*-
"""

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 手写实现灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)  # 将BGR坐标转化为gray坐标并赋值给新图像
print(img)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

# 手写实现二值化
img_binary = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
     for j in range(w):
        if (img_gray[i, j] <= 128):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 225  # 将灰度化坐标转化为二值化坐标

print("image show binary: %s" % img_binary)
cv2.imshow("image show binary", img_binary)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 调接口实现灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 调接口实现二值化
img_binary = np.where(img_gray <= 0.5, 0, 1)
print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

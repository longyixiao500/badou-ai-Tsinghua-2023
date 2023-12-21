#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt


pic_path = r'../file/lenna.png'
img = cv2.imread(pic_path, 1)

# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # ��x��
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # ��y��

# todo �����ݶ�ͼ�񣬼�������x��yͼ�����
# ���ݶ�ͼ��, https://blog.csdn.net/zh_jessica/article/details/77992578
# ͼ�����orͼ���ϼ�Ȩʵ��, cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) �� dst
img_tidu = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)

# Laplace ����
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

# Canny ����
img_canny = cv2.Canny(img_gray, 100, 150)

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(img_tidu,  "gray"), plt.title("img_tidu")
plt.subplot(235), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")
plt.subplot(236), plt.imshow(img_canny, "gray"), plt.title("Canny")
plt.show()



# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : 作业4-1.Histogram Equalization.py
# @Author   : honglin
# @Time     : 2023/5/21 0:04

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取灰度图形
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("image_gray", gray)

# 灰度直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

cv2.waitKey(0)

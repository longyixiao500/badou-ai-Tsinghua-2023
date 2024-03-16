import cv2
# from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

"""图片手工灰度处理"""
# 读取图片
img = cv2.imread("lenna.png")

# 获取图片二维空间宽高
h , w = img.shape[:2]
# 创建单通道空图片，接收读取lenna.png图片的RGB坐标
img_gray = np.zeros([h,w],img.dtype)
# 循环获取lenna.png的RGB坐标
for i in range(h):
    for j in range(w):
        m = img[i,j]
        # 将RGB坐标转化为img_gray的RGB坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

# 打印新图片
print(img_gray)
# 打印新图片坐标
print("image show gray: %s"%img_gray)
# 使用OpenCV的cv2.imshow在窗口显示灰度处理后图像
cv2.imshow("image show gray",img_gray)

# 使用matplotlib.pyplot的subplot函数设置源图像展示坐标系，行数、列数、序号(从左到右，从上到下)
plt.subplot(221)
img = plt.imread("lenna.png")
# 使用matplotlib.pyplot在指定窗口显示源图像
plt.imshow(img)
# 在subplot坐标系中展示opencv 的BGR处理后的灰度图
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

"""图片手工二值化处理"""
rows , clos = img.shape[:2]
img_source = np.zeros([rows,clos],img.dtype)
img_binary = np.zeros([rows,clos],img.dtype)
for i in range(rows):
    for j in range(clos):
        m = img[i,j]
        # 源图像归一化处理
        img_source[i,j] = (m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
        if (img_source[i,j] <= 0.5):
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 1

print(img_source)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
# 显示图片
plt.show()



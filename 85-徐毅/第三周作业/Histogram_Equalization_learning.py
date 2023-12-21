import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("./lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 获得直方图:cv2.calcHist(images, channels, mask, hstiSize, ranges)
# mask:掩模图像，要做出整幅图像的直方图为None
# histSize:取值的间隔范围
# ranges：取值的范围
# gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# plt.subplot(221)
# plt.plot(gray_hist)

plt.subplot(221)
# 做出直方图，与cv2.calcHist二选一
plt.hist(img_gray.ravel(), 256)


# 灰度图像的均衡化cv2.equalizeHist(imgage)
dst = cv2.equalizeHist(img_gray)
# dst_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt.subplot(222)
# plt.plot(dst_hist)

plt.subplot(222)
plt.hist(dst.ravel(), 256)

# 彩色图像均衡化
(b, g, r) = cv2.split(img)
color = ("blue", "green", "red")

# 彩色图像原图的直方图
plt.subplot(223)
for i, color in enumerate(color):
    hist_all = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_all, color = color)
    plt.xlim([0, 256])

# 对每一个通道均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
color = ("blue", "green", "red")

# 彩色图像均衡化后的直方图
plt.subplot(224)
for i, color in enumerate(color):
    hist_all_equ = cv2.calcHist([result], [i], None, [256], [0, 256])
    plt.plot(hist_all_equ, color = color)
    plt.xlim([0, 256])
plt.show()

cv2.imshow("Histogram Equalization0", img)   # 显示原图
cv2.imshow("Histogram Equalization1", img_gray)   # 显示灰度图
cv2.imshow("Histogram Equalization2", dst)   # 显示均衡化后的灰度图
cv2.imshow("Histogram Equalization3", result)   # 显示均衡化后的原图

cv2.waitKey()
cv2.destroyAllWindows()


# new_hist_gray = img_gray.copy()
# height = img_gray.shape[0]
# width = img_gray.shape[1]
# image = height * width
# sum = 0
# for i in range(0, 256):
#     num = 0
#     for x in range(width):
#         for y in range(height):
#             if img_gray[y][x] == i:
#                 num += 1
#     pi = num / image
#     if pi < 0:
#         pi = 0
#     sum += pi
#     q = sum * 256 - 1
#     if q < 0:
#         q = 0
#     new_hist_gray[y, x] = int(q)

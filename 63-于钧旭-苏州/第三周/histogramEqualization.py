import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/lenna.png')
height, width, channel = img.shape
x = np.array([i for i in range(256)])
b = np.zeros_like(x)
g = np.zeros_like(x)
r = np.zeros_like(x)
for i in range(height):
    for j in range(width):
        b[img[i, j, 0]] += 1
        g[img[i, j, 1]] += 1
        r[img[i, j, 2]] += 1
plt.bar(x, b, color='b', label='blue')
plt.bar(x, g, bottom=b, color='g', label='green')
plt.bar(x, r, bottom=b + g, color='r', label='red')
plt.legend()
plt.show()

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
plt.show()

t = height * width
_sum = np.zeros((3), np.float64)
l = np.zeros([256, 3], np.uint8)
for i in range(256):
    _sum[0] += b[i] / t
    _sum[1] += g[i] / t
    _sum[2] += r[i] / t
    l[i] = _sum * 256 - 0.5
    # print(_sum)

    # if b[i]:
    #     _sum[0] += b[i] / t
    #     l[i][0]=_sum[0]* 256 - 0.5
    # if b[i]:
    #     _sum[1] += g[i] / t
    #     l[i][1]=_sum[1]* 256 - 0.5
    # if b[i]:
    #     _sum[2] += r[i] / t
    #     l[i][2]=_sum[2]* 256 - 0.5
# print(l)

img_new = np.zeros_like(img)
for c in range(channel):
    for i in range(height):
        for j in range(width):
            img_new[i, j, c] = l[img[i, j, c], c]

plt.imshow(img_new)
plt.show()


# 彩色图像直方图均衡化
img = cv2.imread('img/lenna.png')
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
for i, col in enumerate(color):
    histr = cv2.calcHist([result], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
plt.show()

cv2.imshow("dst_rgb", result)
cv2.waitKey(0)


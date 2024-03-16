# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : nearest interp.py
# @Author   : honglin
# @Time     : 2023/5/11 21:23

#作业1.实现最临近插值
import cv2;
import numpy as np;


def function(img):
    # print(img.shape)   (512, 512, 3)
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)

    sh = 800 / height
    sw = 800 / width

    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = function(img)

print(zoom)
print("------------------------------------")
print(zoom.shape)

cv2.imshow("old img", zoom)
cv2.imshow("my image", img)

cv2.waitKey(0)

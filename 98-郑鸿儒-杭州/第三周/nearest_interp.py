import cv2
import numpy as np


def zoom_img(img_src, new_width, new_height, flag=False):
    img = cv2.imread(img_src)
    width, height, channels = img.shape
    if flag:
        new_width = int(new_width * width)
        new_height = int(new_height * height)
    new_img = np.zeros((new_width, new_height, channels), np.uint8)

    for i in range(new_width):
        for j in range(new_height):
            x = int(i / new_width * width + 0.5)
            y = int(j / new_height * height + 0.5)
            new_img[i, j] = img[x, y]
    return new_img


zoom = zoom_img('lenna.png', 800, 800)
# zoom1 = zoom_img('lenna.png', 1.5, 1.5, True)

print(zoom)
print(zoom.shape)
# cv2.imshow('nearest interp', zoom1)
cv2.imshow('nearest interp', zoom)
cv2.waitKey()

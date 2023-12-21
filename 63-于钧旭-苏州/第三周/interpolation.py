import cv2
import numpy as np


# 最邻近插值
def nearestInterp(img, h, w):
    height, width, channels = img.shape
    empthImage = np.zeros((abs(h), abs(w), channels), np.uint8)
    sh = h / height
    sw = w / width
    for i in range(abs(h)):
        for j in range(abs(w)):
            x = int(i / sh)
            y = int(j / sw)
            empthImage[i, j] = img[x, y]
    return empthImage


# 双线性插值法
def BiLinear_interpolation(img, H, W):
    src_h, src_w, channels = img.shape
    if src_h == H and src_w == W:
        return img.copy()
    dst_img = np.zeros((abs(H), abs(W), 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / abs(W), float(src_h) / abs(H)
    for i in range(channels):
        for y in range(abs(H)):
            for x in range(abs(W)):
                # 中心重合
                src_x = (x + 0.5) * scale_x - 0.5
                src_y = (y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[y, x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


img = cv2.imread('img/lenna.png')
print(img.shape)
img1 = nearestInterp(img, 400, -600)
print(img1.shape)
cv2.imshow('nearest interpolation', img1)
cv2.waitKey()
img2 = BiLinear_interpolation(img,400,600)
print(img2.shape)
cv2.imshow('blinear interpolation', img2)
cv2.waitKey()
img3=cv2.resize(img,(1000,1000),interpolation=cv2.INTER_AREA)
cv2.imshow('nearest', img3)
cv2.waitKey()

import cv2
import numpy as np

def bilinear_interpolation(img, dst_h, dst_w):
    src_h, src_w, channel = img.shape
    dst_img = np.zeros((dst_h, dst_w, channel), np.uint8)
    ratio_w = float(src_w) / dst_w
    ratio_h = float(src_h) / dst_h
    # print ("src_h, src_w = ", src_h, src_w)
    # print ("dst_h, dst_w = ", dst_h, dst_w)

    # 如果目标图像大小与原图像大小一致，直接输出原图像
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 使两个图像的几何中心重合
                src_x = (dst_x + 0.5) * ratio_w - 0.5
                src_y = (dst_y + 0.5) * ratio_h - 0.5

                # np.floor向下取整，找出双线性插值的四个像素点
                # min：防止取出的原图坐标点超出原图范围，若超出，取src_() - 1
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w -1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 分别找到x方向及y方向的插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # print("temp0, temp1 is:", temp0, temp1)

                # 找到目标点对应原图的坐标
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                # print("dst_img is :", dst_img[dst_y, dst_x, i])

    return dst_img

img = cv2.imread("./lenna.png", 1)
dst_image = bilinear_interpolation(img, 700, 700)

# 直接调用双线性插值函数
dst_image1 = cv2.resize(img, (700,700), interpolation=cv2.INTER_LINEAR)
# print(dst_image.shape)
cv2.imshow('bilinear interp', dst_image)
cv2.imshow('bilinear interp1', dst_image1)
cv2.waitKey()





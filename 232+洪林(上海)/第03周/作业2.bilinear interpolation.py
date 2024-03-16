# 有志者事竟成，破釜沉舟，百二秦关终属楚。
# 苦心人天不负，卧薪尝胆，三千越甲可吞吴。
# @File     : bilinear interpolation.py
# @Author   : honglin
# @Time     : 2023/5/17 23:08

# 作业2.实现双线性插值

import cv2
import numpy as np

'''
python implementation of bilinear interpolation
双线性插值的python实现
'''


def bilinear_interpolation(img, out_dim):
    print(img.shape)
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 中心点对齐
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找出 虚拟点 周围的4个关键坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 套用公式，计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (800, 800))
    cv2.imshow("my bilinear interpolation", dst)
    cv2.waitKey()

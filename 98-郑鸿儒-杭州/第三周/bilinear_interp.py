import cv2
import numpy as np


def bilinear(img, dst_h, dst_w):
    src_h, src_w = img.shape[: 2]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    scale_x = float(src_h) / dst_h
    scale_y = float(src_w) / dst_w
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    for i in range(3):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, src_h - 1)
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, src_w - 1)

                temp0 = (src_y1 - src_y) * img[src_x0, src_y0, i] + (src_y - src_y0) * img[src_x0, src_y1, i]
                temp1 = (src_y1 - src_y) * img[src_x1, src_y0, i] + (src_y - src_y0) * img[src_x1, src_y1, i]
                dst_img[dst_x, dst_y, i] = int((src_x1 - src_x) * temp0 + (src_x - src_x0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # dst = bilinear(img, 700, 700)
    dst = bilinear(img, 1080, 1080)
    cv2.imshow('bilinear_interp', dst)
    cv2.waitKey()

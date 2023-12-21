# -*- coding: utf-8 -*-
"""
@date 2023/4/18
@author: 015643
"""
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


def empty_img(img: ndarray, dst_size):
    return np.zeros((dst_size[0], dst_size[1], img.shape[2]), np.uint8)


def zoom_in_by_nearest(img: ndarray, dst_size):
    """
    最邻近插值实现
    """
    print("zoom_in_by_nearest")
    dst_img_1 = empty_img(img, dst_size)
    # 双线性插值法生成的图像
    start_time = time.time()
    print("开始使用 CV2进行最近插值")
    cv2.resize(img, dst_size, dst_img_1, interpolation=cv2.INTER_NEAREST)
    print("调用CV2耗时：", (time.time() - start_time))

    dst_img_2 = empty_img(img, dst_size)
    start_time = time.time()
    print("开始使用 手工实现最近插值")
    src_y_size, scr_x_size = img.shape[:2]
    hs = dst_size[0] / src_y_size
    ws = dst_size[1] / scr_x_size
    for dst_y in range(dst_size[0]):
        for dst_x in range(dst_size[1]):
            src_x = int(dst_x / ws + 0.5)
            src_y = int(dst_y / hs + 0.5)
            dst_img_2[dst_y, dst_x] = img[src_y, src_x]

    print("手工最近插值耗时：", (time.time() - start_time))

    # dst_img = np.hstack([dst_img_1, dst_img_2])
    # cv2.imshow("CV最邻近-手写最邻近", dst_img)

    plt.figure(figsize=(26, 10))
    plt.subplot(1, 3, 1)
    plt.title("Src")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("CV2ToNearset")
    plt.imshow(cv2.cvtColor(dst_img_1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("M2ToNearset")
    plt.imshow(cv2.cvtColor(dst_img_2, cv2.COLOR_BGR2RGB))
    plt.show()


def zoom_in_by_linear(img: ndarray, dst_size):
    """
    双线性插值实现
    """
    print("zoom_in")
    # 最近插值法生成的图像
    dst_img_0 = empty_img(img, dst_size)
    cv2.resize(img, dst_size, dst_img_0, interpolation=cv2.INTER_NEAREST)

    # 双线性插值法生成的图像
    start_time = time.time()
    print("开始使用 CV2进行双线性插值")
    dst_img_1 = empty_img(img, dst_size)
    cv2.resize(img, dst_size, dst_img_1, interpolation=cv2.INTER_LINEAR)
    print("调用CV2耗时：", (time.time() - start_time))

    # 手写双线性插值法生成的图像
    dst_img_2 = empty_img(img, dst_size)
    src_h, src_w = img.shape[:2]
    hs = dst_size[0] / float(src_h)
    ws = dst_size[1] / float(src_w)

    start_time = time.time()
    print("开始使用 手写双线性插值")
    for dst_z in range(3):
        for dst_y in range(dst_size[0]):
            for dst_x in range(dst_size[1]):
                # 取值第一个点 并增加中心化处理
                # 这里需要注意 课上提供的Demo中 双线性和最邻近的 比例是反着来的。
                # 单线性插值公式推导
                # y  - y0     x -  x0       y  - y0                                     //  假设：     x  - x0
                # -------  =  -------    =>  ------  = @   => y - y0 = (y1 -y0) * @      //       @ = --------
                # y1 - y0     x1 - x0       y1 - y0                                     //            x1 - x0
                #
                # y = (y1 -y0) * @ + y0  => y = (1-@)y0  + y1*@
                # 1 - @ = (x1 - x) / (x1 -x0) -----> 简单不再进行推断
                src_x = (dst_x + 0.5) / ws - 0.5
                src_y = (dst_y + 0.5) / hs - 0.5

                # 可以组合成4个点 (x0, y0) (x1, y0) (x0,y1) (x1, y1)
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                # 计算 (x0, y0)  (x1, y0) 的插值 Q(0)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, dst_z] + (src_x - src_x0) * img[src_y0, src_x1, dst_z]
                # 计算 (x0, y1)  (x1, y1) 的插值 Q(1)
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, dst_z] + (src_x - src_x0) * img[src_y1, src_x1, dst_z]
                # 计算 Q(0) 和 Q(1)的插值
                dst_img_2[dst_y, dst_x, dst_z] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    print("手写双线性插值 耗时：", (time.time() - start_time))
    print("耗时太久，手写方案需要考虑优化~~")
    dst_img = np.hstack([dst_img_0, dst_img_1, dst_img_2])
    print("show image~")
    cv2.imshow("最邻近-CV双线性-手写双线性", dst_img)


def hist(img: ndarray):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)

    plt.figure(num="直方图均衡化", figsize=(20, 10))
    plt.subplot(2, 4, 1)
    plt.title("Src Hist")
    plt.hist(img_gray.ravel(), 256)

    plt.subplot(2, 4, 2)
    plt.title("Src Gray Image")
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.title("Equalize Gray Hist")
    plt.hist(img_hist.ravel(), 256)

    plt.subplot(2, 4, 6)
    plt.title("Equalize Gray Image")
    plt.imshow(img_hist, cmap='gray')

    plt.subplot(2, 4, 3)
    plt.title("Src Image")
    plt.imshow(img_rgb)

    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    dst_rgb_h = cv2.merge((rH, gH, bH))

    plt.subplot(2, 4, 7)
    plt.title("Equalize All Image")
    plt.imshow(dst_rgb_h)

    plt.subplot(2, 4, 4)
    color = ('r', 'g', 'b')
    plt.title("Src RGB")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 4, 8)

    # 三通道均衡化后蓝色变的很多，所以图像变的偏蓝
    plt.title("Equalize RGB")
    for i, col in enumerate(color):
        histr = cv2.calcHist([dst_rgb_h], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.show()


if __name__ == '__main__':
    tips_str = "选择要测试功能：\n1. 最临近差值 \n2. 双线性差值 \n3. 直方图均衡化\n请输入对应的编号"
    select_index = int(input(tips_str))
    img_path = "../res/lenna.png"
    img_original: ndarray = cv2.imread(img_path)
    dsize: tuple = (700, 700)

    if select_index == 1:
        zoom_in_by_nearest(img_original, dsize)
    elif select_index == 2:
        zoom_in_by_linear(img_original, dsize)
        cv2.waitKey(0)
    elif select_index == 3:
        hist(img_original)
    else:
        print("请选择正确的功能")

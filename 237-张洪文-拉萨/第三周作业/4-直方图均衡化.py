# -*- coding: gbk -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt


# ��ͨ��ֱ��ͼ���⻯: equalizeHist
def histogram_equalization_1(img):
    # ��ͼ��תΪ�Ҷ�ͼ��
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.equalizeHist �ú������ܻҶ�ͼ����Ϊ���룬�����ؽ���ֱ��ͼ���⻯���ͼ��
    equalized = cv2.equalizeHist(gray_img)

    # cv2.calcHist()��������һ��ͼ����Ϊ���룬�����ؼ���õ���ֱ��ͼ��
    img_hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # plt.figure()  # ����һ���µ�ͼ�δ��ڣ�������ʾֱ��ͼ
    plt.subplot(121)
    plt.plot(img_hist)

    plt.subplot(122)
    # ʹ��plt.hist()��������ֱ��ͼ, equalized.ravel()����ά����չƽΪһά���飬�Ա���Ϊֱ��ͼ�����롣256��ʾֱ��ͼ��bin���������Ҷ�ֵ��Χ����Ϊ256�����䡣
    plt.hist(equalized.ravel(), 256)
    plt.show()  # ��ʾ���Ƶ�ֱ��ͼͼ�δ���

    # np.hstack([gray_img, equalized])��ԭʼ�Ҷ�ͼ��gray�;�ֱ��ͼ���⻯��ĻҶ�ͼ��dstˮƽ�ѵ���һ��
    cv2.imshow("Histogram Equalization", np.hstack([gray_img, equalized]))
    cv2.waitKey(0)

# ��ͨ��ֱ��ͼ���⻯
def histogram_equalization_2(img):
    # ��ͼ��ת��ΪRGB��ɫ�ռ�
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # �ָ���ɫͨ��
    r, g, b = cv2.split(rgb_img)

    # ��ÿ����ɫͨ������ֱ��ͼ���⻯
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # �ϲ����⻯�����ɫͨ��
    equalized_img = cv2.merge([r_eq, g_eq, b_eq])

    # ��ʾԭʼͼ��;��⻯���ͼ��
    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(equalized_img)
    plt.title("Equalized Image")

    plt.show()


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # histogram_equalization_1(img)
    histogram_equalization_2(img)

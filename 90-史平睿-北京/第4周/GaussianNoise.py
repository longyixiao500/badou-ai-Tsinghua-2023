
import cv2 as cv
import numpy as np
from numpy import shape
import random

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		#ÿ��ȡһ�������
		#��һ��ͼƬ���������к��б�ʾ�Ļ���randX ����������ɵ��У�randY����������ɵ���
        #random.randint�����������
		#��˹����ͼƬ��Ե��������-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #�˴���ԭ�����ػҶ�ֵ�ϼ��������
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #���Ҷ�ֵС��0��ǿ��Ϊ0�����Ҷ�ֵ����255��ǿ��Ϊ255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv.imread("lenna.png", 0)
img1 = GaussianNoise(img, 2, 4, 0.3)
cv.imshow("lenna_GaussianNoise", img1)

img = cv.imread('lenna.png')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("source", img2)
cv.waitKey(0)
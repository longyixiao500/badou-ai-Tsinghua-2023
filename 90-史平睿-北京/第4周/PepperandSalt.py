
import cv2 as cv
import numpy as np
from numpy import shape
import random

def fun1(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		#ÿ��ȡһ�������
		#��һ��ͼƬ���������к��б�ʾ�Ļ���randX ����������ɵ��У�randY����������ɵ���
        #random.randint�����������
		#��������ͼƬ��Ե��������-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #random.random��������������������յ�һ�����ص���һ��Ŀ����ǰ׵�255��һ��Ŀ����Ǻڵ�0
        if  random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv.imread('lenna.png', 0)
img1 = fun1(img,0.3)
#���ļ�����д������Ϊlenna_PepperandSalt.png�ļ�����ͼƬ
cv.imwrite('lenna_PepperandSalt.png',img1)
cv.imshow("lenna_PepperandSalt", img1)

img = cv.imread('lenna.png')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("source", img2)
cv.waitKey(0)
import cv2
import matplotlib.pyplot as plt
import numpy as np

def grayHist(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    IMG = cv2.equalizeHist(gray)
    cv2.imshow('Hist',np.hstack([gray,IMG]))
    cv2.waitKey(0)

def BGRHist(img):
    (b,g,r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    IMG = cv2.merge((bH,gH,rH))
    cv2.imshow('BGRHist',np.hstack([img,IMG]))
    cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    x = input('输入你想选择的直方图均衡化模式：')
    if x == '灰度':
        grayHist(img)
    if x == '彩色':
        BGRHist(img)
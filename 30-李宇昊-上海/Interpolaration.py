import cv2
import numpy as np

def nearest(img,h,w):
    H,W,channel = img.shape
    emptyImg = np.zeros((h,w,channel),np.uint8)
    sh = h/H
    sw = w/W
    for i in range(h):
        for j in range(w):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            emptyImg[i,j] = img[x,y]
    return emptyImg

def biliner(img,h,w):
    H,W,channel = img.shape
    if h ==H and w == W:
        return img.copy()
    IMAGE = np.zeros((h,w,channel),dtype=np.uint8)
    sh,sw = float(H)/float(h),float(W)/float(w)
    for i in range(channel):
        for y in range(h):
            for x in range(w):
                X = (x+0.5)*sw-0.5
                Y = (y+0.5)*sh-0.5

                X0 = int(np.floor(X))
                X1 = min(X0 + 1,W - 1)
                Y0 = int(np.floor(Y))
                Y1 = min(Y0 + 1,H - 1)

                t0 = (X1 - X)*img[Y0,X0,i]+(X - X0)*img[Y0,X1,i]
                t1 = (X1 - X)*img[Y1,X0,i]+(X - X0)*img[Y1,X1,i]
                IMAGE[y,x,i] = int((Y1-Y)*t0 + (Y-Y0)*t1)
    return IMAGE

if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    x = input('请输入你希望使用的插值算法：')
    h,w = (input('请输入你希望得到图片的大小：').split())
    h = int(h)
    w = int(w)
    if x == '最邻近':
        Result = nearest(img,h,w)
    if x == '双线性':
        Result = biliner(img,h,w)
    cv2.imshow('lenna',img)
    cv2.imshow('Interpolaration result',Result)
    cv2.waitKey(0)
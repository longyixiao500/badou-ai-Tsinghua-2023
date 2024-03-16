import cv2
import numpy as np
import matplotlib.pyplot as plt
from img2gray import  img2gray_self

def img2binary_self(image,thres):
    assert image is not None,"not find image!!!"
    gray = img2gray_self(image)
    #灰度化后遍历，大于一个数取255，小于取0
    binary = np.zeros_like(gray)
    w,h = gray.shape[:2]
    for i in range(w):
        for j in range(h):
            if gray[i,j] > thres:
                binary[i,j] =255
            else:
                binary[i,j] = 0
    return binary
#调用接口实现
def img2binary(image):
    assert image is not None,"not find image!!!"
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    thread,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return binary

def main():
    image = cv2.imread("dog.jpg")
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #画自己写的转二值图
    binary_self = img2binary_self(img,127)
    plt.imshow(binary_self,cmap="gray")
    plt.show()
    #画cv2接口的二值图
    binary = img2binary(img)
    cv2.imshow("binary_dog",binary)
    cv2.waitKey(2000)


if __name__ == '__main__':
    main()
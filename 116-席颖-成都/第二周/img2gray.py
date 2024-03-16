import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#自己手动实现，rgb彩色图片转成灰度图
def img2gray_self(image):
    assert image is not None,"not find image!!!"
    h,w,c = image.shape[:3]
    print(h,w,c)
    #创建同样大小的单通道图片
    img_gray = np.zeros((h,w,1),np.uint8)
    for i in range(h):
        for j  in range(w):
            tmp = image[i,j,0]*0.3 + image[i,j,1]*0.59 + image[i,j,2]*0.11
            img_gray[i,j] = np.uint(tmp)
    return img_gray

#调用接口实现 rgb彩色图片转灰度图
def img2gary(image):
    assert image is not None,"not find image!!!"
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return img_gray

def main():
    image_bgr = cv2.imread("dog.jpg")
    #cv2.imshow("dog",image_bgr)
    #bgr 转成rgb  cv2.imshow 处理的是bgr格式 ，plt.show处理的是rgb格式
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    gary_self = img2gray_self(image_rgb)
    plt.imshow(gary_self,cmap="gray")
    plt.show()
    #画出cv2函数实现的
    gary = img2gary(image_rgb)
    plt.imshow(gary,cmap="gray")
    plt.show()
    #cv2.imshow("gray_dog",gary)
    #cv2.waitKey(5000)


if __name__ == "__main__":
    main()
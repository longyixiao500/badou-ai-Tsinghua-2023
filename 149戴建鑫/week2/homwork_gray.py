import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.color import rgb2gray

#算法显示部分
#灰度化
img=cv2.imread("lenna.jpg")#读取原始图像
h,w=img.shape[0:2]#获取前两个索引的值
img_gray=np.zeros([h,w],img.dtype)#np创建空的等同类型与尺寸的空图
for i in range(h):
  for j in range(w):
    m_data=img[i,j]
    img_gray[i,j]=int(m_data[0]*0.11+ m_data[1]*0.59 + m_data[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像 应为opencv读取的图像是BGR类型
    
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

#接口调用
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#调用opencv的方法
cv2.imshow("image show gray with opencv2",img_gray2)

plt.subplot(221)
img = plt.imread("lenna.png") 
plt.imshow(img)
print(img)
img_gray3 = rgb2gray(img)#通过skimage实现
plt.subplot(222)
plt.imshow(img_gray3), cmap='gray')
print(img_gray3)


#二值化算法实现
img_binary = np.zeros([h,w],img_gray.dtype)
rows, cols = img_binary.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 127):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')


#接口实现二值化
ret, img_binary2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.subplot(234)
plt.imshow(img_binary2, cmap='binary')
plt.show()

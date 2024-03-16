
import cv2
import numpy as np
def function(img, out_dim):
    height, width, channels = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    emptyImage = np.zeros((dst_h, dst_w, channels), np.uint8)
    sh=dst_h/height
    sw=dst_w/width
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage
    
# cv2.resize(img, (800,800,c),near/bin)

img=cv2.imread("lenna.png")
zoom=function(img, (800, 800))
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)



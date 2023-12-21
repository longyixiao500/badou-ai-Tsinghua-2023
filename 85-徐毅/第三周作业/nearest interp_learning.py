import cv2
import numpy as np

def neareast_interp(img, dst_height, dst_width):
    """

    :param img: 输入的图像
    :param new_height: 缩放后图像的高
    :param new_width: 缩放后图像的宽
    :return:返回缩放后的图像
    """

    # 取出原图的shape
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    new_image = np.zeros((dst_height, dst_width, channels), np.uint8)

    # 缩放比例
    ratio_h = dst_height / height
    ratio_w = dst_width / width

    # 遍历缩放图像对应原图的像素点
    for i in range(3):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                src_x = int(dst_x / ratio_w + 0.5)
                src_y = int(dst_y / ratio_h + 0.5)

                # 防止对应原图像素点超出范围
                src_x0 = min(src_x, width - 1)
                src_y0 = min(src_y, height - 1)
                new_image[dst_y, dst_x] = img[src_y0, src_x0]
                # print(img[src_y, src_x])
    return new_image

# 直接调用最邻近插值函数
# dst0 = cv2.resize(img, (300,200), interpolation=cv2.INTER_NEAREST)

img = cv2.imread("./learning.png")
new_img =neareast_interp(img, 700, 700)
# new_img1 =neareast_interp(img, 400, 300)
# new_img2 =neareast_interp(img, 300, 400)
print(new_img.shape)
# print(img[511, 511])
cv2.imshow("nearest interp1", new_img)
# cv2.imshow("nearest interp2", new_img1)
# cv2.imshow("nearest interp3", new_img2)
cv2.waitKey(0)



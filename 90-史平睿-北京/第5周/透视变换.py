import cv2
import numpy as np

img = cv2.imread('photo.jpg')

result3 = img.copy()

'''
ע������src��dst�����벢����ͼ�񣬶���ͼ���Ӧ�Ķ������ꡣ
'''
#src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
#dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
src = np.float32([[3, 50], [400, 40], [30, 260], [385, 270]])
dst = np.float32([[0, 0], [370, 0], [0, 220], [370, 220]])
print(img.shape)
# ����͸�ӱ任���󣻽���͸�ӱ任
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (370, 220))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)

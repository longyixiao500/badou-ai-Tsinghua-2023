import cv2
import numpy as np

#��ֵ��ϣ�㷨
def aHash(img):
    img = cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    avg = s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#��ֵ��ϣ�㷨
def dHash(img):
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1
    return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_PepperandSalt.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print("��ֵ��ϣ�㷨���ƶȣ�", n)

hash2 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1,hash2)
print("��ֵ��ϣ�㷨���ƶȣ�", n)
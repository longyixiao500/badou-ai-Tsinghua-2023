import numpy as np
import matplotlib.pyplot as plt
import math
 
if __name__ == '__main__':
    pic_path = 'lenna.png' 
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .pngͼƬ������Ĵ洢��ʽ��0��1�ĸ�����������Ҫ��չ��255�ټ���
        img = img * 255  # ���Ǹ���������
    img = img.mean(axis=-1)  # ȡ��ֵ���ǻҶȻ���
 
    # 1����˹ƽ��
    #sigma = 1.52  # ��˹ƽ��ʱ�ĸ�˹�˲�������׼��ɵ�
    sigma = 0.5  # ��˹ƽ��ʱ�ĸ�˹�˲�������׼��ɵ�
    dim = int(np.round(6 * sigma + 1))  # round���������뺯�������ݱ�׼�����˹���Ǽ��˼��ģ�Ҳ����ά��
    if dim % 2 == 0:  # ���������,���ǵĻ���һ
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # �洢��˹�ˣ��������鲻���б���
    tmp = [i-dim//2 for i in range(dim)]  # ����һ������
    n1 = 1/(2*math.pi*sigma**2)  # �����˹��
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # �洢ƽ��֮���ͼ��zeros�����õ����Ǹ���������
    tmp = dim//2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # ��Ե�
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # ��ʱ��img_new��255�ĸ��������ݣ�ǿ������ת���ſ��ԣ�gray�ҽ�
    plt.axis('off')
 
    # 2�����ݶȡ������������˲����ݶ��õ�sobel���󣨼��ͼ���е�ˮƽ����ֱ�ͶԽǱ�Ե��
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # �洢�ݶ�ͼ��
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # ��Ե��������������ṹ����д1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x����
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y����
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y/img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
 
    # 3���Ǽ���ֵ����
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # ��8�������Ƿ�ҪĨȥ�������
            temp = img_tidu[i-1:i+2, j-1:j+2]  # �ݶȷ�ֵ��8�������
            if angle[i, j] <= -1:  # ʹ�����Բ�ֵ���ж��������
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
 
    # 4��˫��ֵ��⣬���ӱ�Ե����������һ���Ǳߵĵ�,�鿴8�����Ƿ�����п����Ǳߵĵ㣬��ջ
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # ���������ø���ֵ�ǵ���ֵ������
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # ��Ȧ��������
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:  # ȡ��һ���Ǳߵĵ�
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # ��
                img_yizhi[i, j] = 0
 
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # ��ջ
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1-1, temp_2-1] = 255  # ������ص���Ϊ��Ե
            zhan.append([temp_1-1, temp_2-1])  # ��ջ
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
 
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
 
    # ��ͼ
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # �ر�����̶�ֵ
    plt.show()

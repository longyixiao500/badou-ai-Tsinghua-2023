"""
ʹ��PCA����������X��K�׽�ά����Z
"""

import numpy as np

class CPCA(object):
    '''
    ��PCA����������X��K�׽�ά����Z
    Note:�뱣֤�������������X shape=(m,n),m��������n������
    '''
    def __init__(self, X, K):
        '''
        :param X,ѵ����������X
        :param K,X�Ľ�ά����Ľ�������XҪ������ά��k��
        '''
        self.X = X       #��������X
        self.K = K       #K�׽�ά�����Kֵ
        self.centrX = [] #����X�����Ļ�
        self.C = []      #��������Э�������C
        self.U = []      #��������X�Ľ�άת������
        self.Z = []      #��������X�Ľ�ά����Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU���

    def _centralized(self):
        '''����X�����Ļ�'''
        print('��������X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #��������������ֵ
        print('��������������ֵ:\n',mean)
        centrX = self.X - mean ##�����������Ļ�
        print('��������X�����Ļ�centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''����������X��Э�������C'''
        #����������������
        ns = np.shape(self.centrX)[0]
        #���������Э�������C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('��������X��Э�������C:\n', C)
        return C

    def _U(self):
        '''��X�Ľ�άת������U, shape=(n,k), n��X������ά��������k�ǽ�ά���������ά��'''
        #����X��Э�������C������ֵ����������
        a,b = np.linalg.eig(self.C) #����ֵ��ֵ��a����Ӧ����������ֵ��b��
        print('��������Э�������C������ֵ:\n', a)
        print('��������Э�������C����������:\n', b)
        #��������ֵ�����topK����������
        ind = np.argsort(-1*a)
        #����K�׽�ά�Ľ�άת������U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d�׽�άת������U:\n'%self.K, U)
        return U
        
    def _Z(self):
        '''����Z=XU��ά����Z, shape=(m,k), n������������k�ǽ�ά����������ά������'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('��������X�Ľ�ά����Z:\n', Z)
        return Z

if __name__=='__main__':
    '10����3������������, ��Ϊ��������Ϊ����ά��'
    X = np.array([[10, 15, 29],
                    [15, 46, 13],
                    [23, 21, 30],
                    [11, 9,  35],
                    [42, 45, 11],
                    [9,  48, 5],
                    [11, 21, 14],
                    [8,  5,  15],
                    [11, 12, 21],
                    [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('������(10��3�У�10��������ÿ������3������):\n', X)
    pca = CPCA(X,K)
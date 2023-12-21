#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
 
#print(tf.__version__)

#ʹ��numpy����200�������
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
 
#��������placeholder�����������
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
 
#�����������м��
Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))    #����ƫ����
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)   #���뼤���

#���������������
Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))  #����ƫ����
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)   #���뼤���

#������ʧ���������������
loss=tf.reduce_mean(tf.square(y-prediction))
#���巴�򴫲��㷨��ʹ���ݶ��½��㷨ѵ����
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
    #������ʼ��
    sess.run(tf.global_variables_initializer())
    #ѵ��2000��
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
 
    #���Ԥ��ֵ
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
 
    #��ͼ
    plt.figure()
    plt.scatter(x_data,y_data)   #ɢ������ʵֵ
    plt.plot(x_data,prediction_value,'r-',lw=5)   #������Ԥ��ֵ
    plt.show()

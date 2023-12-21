import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
w_l1 = tf.Variable(tf.random.normal([1, 10]))
b_l1 = tf.Variable(tf.zeros([1, 10]))   # 加入偏置项
wx_plus_b_l1 = tf.matmul(x, w_l1) + b_l1
l1 = tf.nn.tanh(wx_plus_b_l1)   # 加入激活函数

# 定义神经网络输出层
w_l2 = tf.Variable(tf.random.normal([10, 1]))
b_l2 = tf.Variable(tf.zeros([1, 1]))    # 偏置项
wx_plus_b_l2 = tf.matmul(l1, w_l2) + b_l2
prediction = tf.nn.tanh(wx_plus_b_l2)   # 激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))

# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获取预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)     # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)      # 曲线是预测值
    plt.show()

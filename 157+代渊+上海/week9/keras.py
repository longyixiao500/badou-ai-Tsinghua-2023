from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# 1.将训练数据和检测数据加载到内存中
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

# 2.把用于测试的第一张图片打印出来看看
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 3.使用tensorflow.Keras搭建一个有效识别图案的神经网络，
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# 4.把数据做归一化处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 5.把图片对应的标记也做一个更改
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


# 6.开始训练
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

# 7.测试数据输入，检验网络学习后的图片识别效果.
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss) 
print('test_acc', test_acc)

# 8.输入一张手写数字图片到网络中，看看它的识别效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break


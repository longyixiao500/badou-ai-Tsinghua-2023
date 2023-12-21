from nets import vgg16
import tensorflow.compat.v1 as tf
import numpy as np
import utils

# ��ȡͼƬ
img1 = utils.load_image("./test_data/table.jpg")

# �������ͼƬ����resize��ʹ��shape����(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_image(inputs, (224, 224))

# ��������ṹ
prediction = vgg16.vgg_16(resized_img)

# ����ģ��
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# ���������softmaxԤ��
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

# ��ӡԤ����
print("result: ")
utils.print_prob(pre[0], './synset.txt')

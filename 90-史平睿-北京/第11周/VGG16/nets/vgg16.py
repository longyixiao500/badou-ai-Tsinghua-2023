#-------------------------------------------------------------#
#   vgg16�����粿��
#-------------------------------------------------------------#
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()

import tf_slim as slim

# ����slim����
#slim = tf.contrib.slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # ����vgg_16������

        # conv1����[3,3]������磬�����������Ϊ64�����Ϊ(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2���ػ������netΪ(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2����[3,3]������磬�����������Ϊ128�����netΪ(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2���ػ������netΪ(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3����[3,3]������磬�����������Ϊ256�����netΪ(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2���ػ������netΪ(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3����[3,3]������磬�����������Ϊ256�����netΪ(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2���ػ������netΪ(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3����[3,3]������磬�����������Ϊ256�����netΪ(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2���ػ������netΪ(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # ���þ���ķ�ʽģ��ȫ���Ӳ㣬Ч����ͬ�����netΪ(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                            scope='dropout6')
        # ���þ���ķ�ʽģ��ȫ���Ӳ㣬Ч����ͬ�����netΪ(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                            scope='dropout7')
        # ���þ���ķ�ʽģ��ȫ���Ӳ㣬Ч����ͬ�����netΪ(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
        
        # �����þ���ķ�ʽģ��ȫ���Ӳ㣬���������Ҫƽ��
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Input, ZeroPadding2D, AveragePooling2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # conv + batchNorm + relu
    x = Conv2D(filters=filters1,
               kernel_size=(1, 1),
               strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    # (x)是执行了__call__自调用的入参
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # conv + batchNorm + relu
    x = Conv2D(filters=filters2,
               kernel_size=kernel_size,
               padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # conv + batchNorm
    x = Conv2D(filters=filters3,
               kernel_size=(1, 1),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # shortcut
    shortcut = Conv2D(filters=filters3,
                      kernel_size=(1, 1),
                      strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # conv + batchNorm + relu
    x = Conv2D(filters=filters1,
               kernel_size=(1, 1),
               name=conv_name_base + '2a')(input_tensor)
    # (x)是执行了__call__自调用的入参
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # conv + batchNorm + relu
    x = Conv2D(filters=filters2,
               kernel_size=kernel_size,
               padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # conv + batchNorm
    x = Conv2D(filters=filters3,
               kernel_size=(1, 1),
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


"""
构建模型结构
"""
def Resnet50(input_shape=(224, 224, 3), classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, (3, 3), [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='b')
    x = identity_block(x, (3, 3), [64, 64, 256], stage=2, block='c')

    x = conv_block(x, (3, 3), [128, 128, 512], stage=3, block='a')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='b')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='c')
    x = identity_block(x, (3, 3), [128, 128, 512], stage=3, block='d')

    x = conv_block(x, (3, 3), [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, (3, 3), [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, (3, 3), [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, (3, 3), [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = Resnet50()
    model.summary()
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))



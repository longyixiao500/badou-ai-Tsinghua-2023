#���ļ������ȡCifar-10���ݲ��������������ǿԤ����
import os
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.disable_eager_execution()
num_classes=10

#�趨����ѵ������������������
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#����һ�����࣬���ڷ��ض�ȡ��Cifar-10������
class CIFAR10Record(object):
    pass


#����һ����ȡCifar-10�ĺ���read_cifar10()�����������Ŀ�ľ��Ƕ�ȡĿ���ļ����������
def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1                                            #�����Cifar-100���ݼ�����˴�Ϊ2
    result.height=32
    result.width=32
    result.depth=3                                           #��Ϊ��RGB��ͨ�������������3

    image_bytes=result.height * result.width * result.depth  #ͼƬ������Ԫ������
    record_bytes=label_bytes + image_bytes                   #��Ϊÿһ����������ͼƬ�ͱ�ǩ���������յ�Ԫ����������ҪͼƬ������������һ����ǩֵ

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)  #ʹ��tf.FixedLengthRecordReader()����һ���ļ���ȡ�ࡣ�����Ŀ�ľ��Ƕ�ȡ�ļ�
    result.key,value=reader.read(file_queue)                 #ʹ�ø����read()�������ļ����������ȡ�ļ�

    record_bytes=tf.decode_raw(value,tf.uint8)               #��ȡ���ļ��Ժ󣬽���ȡ�����ļ����ݴ��ַ�����ʽ����Ϊͼ���Ӧ����������
    
    #��Ϊ�������һ��Ԫ���Ǳ�ǩ����������ʹ��strided_slice()��������ǩ��ȡ����������ʹ��tf.cast()��������һ����ǩת����int32����ֵ��ʽ
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    #ʣ�µ�Ԫ���ٷָ��������Щ����ͼƬ���ݣ���Ϊ��Щ���������ݼ�����洢����ʽ��depth * height * width������Ҫ�����ָ�ʽת����[depth,height,width]
    #��һ���ǽ�һά����ת����3ά����
    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                           [result.depth,result.height,result.width])  

    #����Ҫ��֮ǰ�ָ�õ�ͼƬ����ʹ��tf.transpose()����ת����Ϊ�߶���Ϣ�������Ϣ�������Ϣ������˳��
    #��һ����ת�������Ų���ʽ����Ϊ(h,w,c)
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result                                 #����ֵ���Ѿ���Ŀ���ļ��������Ϣ����ȡ����

def inputs(data_dir,batch_size,distorted):               #��������Ͷ����ݽ���Ԥ����---��ͼ�������Ƿ������ǿ�����жϣ���������Ӧ�Ĳ���
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]   #ƴ�ӵ�ַ

    file_queue=tf.train.string_input_producer(filenames)     #�����Ѿ��е��ļ���ַ����һ���ļ�����
    read_input=read_cifar10(file_queue)                      #�����Ѿ��е��ļ�����ʹ���Ѿ�����õ��ļ���ȡ����read_cifar10()��ȡ�����е��ļ�

    reshaped_image=tf.cast(read_input.uint8image,tf.float32)   #���Ѿ�ת���õ�ͼƬ�����ٴ�ת��Ϊfloat32����ʽ

    num_examples_per_epoch=num_examples_pre_epoch_for_train


    if distorted != None:                         #���Ԥ�������е�distorted������Ϊ��ֵ���ʹ���Ҫ����ͼƬ��ǿ����
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])          #���Ƚ�Ԥ����õ�ͼƬ���м��У�ʹ��tf.random_crop()����

        flipped_image=tf.image.random_flip_left_right(cropped_image)    #�����кõ�ͼƬ�������ҷ�ת��ʹ��tf.image.random_flip_left_right()����

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)   #�����ҷ�ת�õ�ͼƬ����������ȵ�����ʹ��tf.image.random_brightness()����

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)    #�����ȵ����õ�ͼƬ��������Աȶȵ�����ʹ��tf.image.random_contrast()����

        float_image=tf.image.per_image_standardization(adjusted_contrast)          #���б�׼��ͼƬ������tf.image.per_image_standardization()�����Ƕ�ÿһ�����ؼ�ȥƽ��ֵ���������ط���

        float_image.set_shape([24,24,3])                      #����ͼƬ���ݼ���ǩ����״
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
                             #ʹ��tf.train.shuffle_batch()�����������һ��batch��image��label

        return images_train,tf.reshape(labels_train,[batch_size])

    else:                               #����ͼ�����ݽ���������ǿ����
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #����������£�ʹ�ú���tf.image.resize_image_with_crop_or_pad()��ͼƬ���ݽ��м���

        float_image=tf.image.per_image_standardization(resized_image)          #��������Ժ�ֱ�ӽ���ͼƬ��׼������

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #����ʹ��batch()��������tf.train.shuffle_batch()����
        return images_test,tf.reshape(labels_test,[batch_size])

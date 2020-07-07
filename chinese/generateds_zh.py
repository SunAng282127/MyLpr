# !/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import codecs
import os

image_train_path = './data_jpg/train_jpg/'
label_train_path = './data_jpg/train_label.txt'
tfRecord_train = './data/train.tfrecords'
image_test_path = './data_jpg/test_jpg/'
label_test_path = './data_jpg/test_label.txt'
tfRecord_test = './data/test.tfrecords'
data_path = './data'
z_Chart = './data_jpg/z_Chart.txt'
resize_height = 20
resize_width = 20


# 遍历对照表
def traversal_chart(chart):
    f = codecs.open(z_Chart, 'r', 'utf-8')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        if value[1] == chart:
            return int(value[0])


# 创建写入的tfrecord
def write_tfRecord(tfRecordName, image_path, label_path):
    # 新建一个writer
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = codecs.open(label_path, 'r', 'utf-8')
    contents = f.readlines()
    f.close()

    # 循环遍历每一张图和标签
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        print(img_path)
        img = Image.open(img_path).convert('L')
        img_raw = img.tobytes()
        # 将字母或数字换为对应的十进制数字
        print(value[1])
        num_char = traversal_chart(value[1])
        labels = [0] * 31
        labels[num_char] = 1
        print(labels)
        # tf.train.Example用来存储训练数据，训练数据的特征用键值对的形式表示,把每张图片和标签封装到example中
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        # 把数据序列化成字符串存储
        writer.write(example.SerializeToString())
        num_pic += 1
        print('the number of picture:', num_pic)
    writer.close()
    print('write tfrecord successful')


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecord_path):
    # 该函数会生成一个先入先出的队列，文件阅读器会使用它来读取数据
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    # 创建一个reader来读取tfRecord文件中的样例
    reader = tf.TFRecordReader()
    # 解析读入的一个样例
    _, serialized_example = reader.read(filename_queue)
    # 把读出的每个样本保存在serialized_example中进行解序列化，利用FixedLenFeature解析得到一个张量
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([31], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    # 将img_raw字符串转换8位无符号整数，利用decode_raw函数将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 将形状变为一行400列
    img.set_shape([400])
    # 变为0-1之间的浮点数
    img = tf.cast(img, tf.float32) * (1. / 255)
    # 把标签列表变为浮点数
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    # 随机读取一个batch的数据
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=500)
    return img_batch, label_batch


def keras_read(num, isTrain):
    img_batch, label_batch = get_tfrecord(num, isTrain)
    # 创建一个会话，并通过python中的上下文管理器来管理这个会话
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # 开启线程协调器
        coord = tf.train.Coordinator()
        # 启动输入队列的线程，填充训练样本到队列中，以便出队操作可以从队列中拿到样本
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        xs, ys = sess.run([img_batch, label_batch])
        # 关闭线程协调器
        coord.request_stop()
        coord.join(threads)
    return xs, ys


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()

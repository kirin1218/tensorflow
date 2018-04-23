# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)

train_images, train_labels = mnist.train.next_batch(50)


test_images = mnist.test.images

test_labels = mnist.test.labels

#入力データの宣言
x = tf.placeholder(tf.float32, [None, 784])

#入力層から中間層
w_1 = tf.Variable(tf.truncated_normal([784,64], stddev=0.1), name="w1")
b_1 = tf.Variable(tf.zeros([64]), name="b1")
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

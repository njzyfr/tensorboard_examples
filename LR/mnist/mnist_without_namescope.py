# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = '/tmp/tensorboard_examples/lr/mnist/mnist_without_namescope'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 类对象（train, validation, test）

def train(batch_size=100, lr=0.5, iter_num=1000):
    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=(None, 28 * 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    w = tf.Variable(tf.truncated_normal(shape=(28 * 28, 10), stddev=0.5))
    b = tf.Variable(tf.zeros(shape=[10]))

    Wx_plus_b = tf.matmul(x, w) + b
    probability = tf.nn.softmax(Wx_plus_b)

    loss = -tf.reduce_mean(y * tf.log(probability))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss=loss)

    comparision = tf.equal(tf.argmax(probability, dimension=1), tf.argmax(y, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(comparision, dtype=tf.float32))

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)

    for i in range(iter_num):
        batch = mnist.train.next_batch(batch_size=batch_size)
        sess.run(trainer, feed_dict={x: batch[0], y: batch[1]})

    train_writer.close()
    sess.close()

if __name__ == '__main__':
    train()
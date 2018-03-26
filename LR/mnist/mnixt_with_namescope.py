# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = '/tmp/tensorboard_examples/lr/mnist/mnist_with_namescope'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 类对象（train, validation, test）

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def train(batch_size=100, lr=0.5, iter_num=1000):
    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=(None, 28 * 28), name="x")
    y = tf.placeholder(tf.float32, shape=(None, 10), name="y")

    with tf.name_scope('weights'):
        w = tf.Variable(tf.truncated_normal(shape=(28 * 28, 10), stddev=0.5))
        variable_summaries(w)

    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros(shape=[10]))
        variable_summaries(b)


    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(x, w) + b
        tf.summary.histogram('Wx_plus_b', Wx_plus_b)

    probability = tf.nn.softmax(Wx_plus_b, name='probability')
    tf.summary.histogram('probability', probability)

    with tf.name_scope("loss"):
        loss = -tf.reduce_mean(y * tf.log(probability))
        tf.summary.scalar('loss', loss)

    with tf.name_scope("trainer"):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss=loss)

    with tf.name_scope("accuracy"):
        comparision = tf.equal(tf.argmax(probability, dimension=1), tf.argmax(y, dimension=1))
        accuracy = tf.reduce_mean(tf.cast(comparision, dtype=tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOGDIR + '/test')

    for i in range(iter_num):
        batch = mnist.train.next_batch(batch_size=batch_size)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch[0], y: batch[1]})
            train_writer.add_summary(s, i)
            [test_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: mnist.test.images, y: mnist.test.labels})
            test_writer.add_summary(s, i)
        sess.run(trainer, feed_dict={x: batch[0], y: batch[1]})

    train_writer.close()
    test_writer.close()
    sess.close()

if __name__ == '__main__':
    train()
import numpy as np
from pickle import dump as pdump
from pickle import load as pload
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def run_1():
    # print('loading data...')
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # pdump(mnist, open('mnist.dat', 'wb'))

    mnist = pload(open('mnist.dat', 'rb'))

    print('instantiating variables...')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    print('training...')
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(1000):
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    print('check accuracy...')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    return 0


if __name__ == '__main__':
    run_1()

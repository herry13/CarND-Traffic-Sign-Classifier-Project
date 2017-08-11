#!/usr/bin/env python3

import pickle
import argparse
from collections import namedtuple
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return data['features'], data['labels']


class SignNet:
    """
    SignNet: a CNN model for Traffic Sign recognition.

    TODO: save/restore the model to/from file.
    """
    def __init__(self, args):
        with tf.variable_scope(args.name):
            self.scope = args.name
            
            # Variables
            x_shape = (None, args.shape[0], args.shape[1], args.shape[2])
            self.x = tf.placeholder(tf.float32, x_shape, name='x')
            self.y = tf.placeholder(tf.int32, (None), name='y')
            one_hot_y = tf.one_hot(self.y, args.num_classes)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # Convolution layers
            self.conv1 = self._convolution(1, self.x, args.conv1, 2, 2)
            self.conv2 = self._convolution(2, self.conv1, args.conv2, 4, 4)

            # Combine conv1 and conv2
            fc0_1 = flatten(self.conv1)
            fc0_2 = flatten(self.conv2)
            fc0 = tf.concat([fc0_1, fc0_2], 1)

            # Fully-connected layers
            fc1 = self._fully_connected(1, fc0, args.fc1)
            fc2 = self._fully_connected(2, fc1, args.fc2)
            self.logits = self._logits(fc2, args.num_classes)

            # Operations
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=one_hot_y, logits=self.logits,
                    name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy, name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            self.train_op = optimizer.minimize(self.loss, name='train')
            self.correct = tf.equal(tf.argmax(self.logits, 1),
                    tf.argmax(one_hot_y, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(self.correct, tf.float32),
                    name='accuracy')

    
    def _logits(self, x, size):
        shape = (int(x.shape[1]), size)
        weights = tf.Variable(self._tf_var(shape), name='logits_W')
        biases = tf.Variable(tf.zeros(size), name='logits_b')
        logits = tf.add(tf.matmul(x, weights), biases, name='logits')
        return logits

    def _tf_var(self, shape):
        return tf.truncated_normal(shape=shape, mean=0, stddev=0.1)


    def _fully_connected(self, idx, x, size):
        name, shape = 'fc{}_W'.format(idx), (int(x.shape[1]), size)
        weights = tf.Variable(self._tf_var(shape), name=name)
        name = 'fc{}_b'.format(idx)
        biases = tf.Variable(tf.zeros(size), name=name)
        fc = tf.matmul(x, weights) + biases
        fc = tf.nn.relu(fc)
        name = 'fc{}'.format(idx)
        fc = tf.nn.dropout(fc, keep_prob=self.keep_prob, name=name)
        return fc
        

    def _convolution(self, idx, x, features, kernelsize, poolsize):
        # Convolution
        name = 'conv{}_W'.format(idx)
        shape = (kernelsize, kernelsize, int(x.shape[3]), features)
        weights = tf.Variable(self._tf_var(shape), name=name)
        name = 'conv{}_b'.format(idx)
        biases = tf.Variable(tf.zeros(features), name=name)
        conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') \
                    + biases
        # Activation
        conv = tf.nn.relu(conv)
        # Dropout
        conv = tf.nn.dropout(conv, keep_prob=self.keep_prob)
        # Max Pooling
        name = 'conv{}'.format(idx)
        conv = tf.nn.max_pool(conv, ksize=[1, poolsize, poolsize, 1], \
                              strides=[1, poolsize, poolsize, 1], \
                              padding='VALID', name=name)
        return conv

    
    '''def t(name):
        return '{}/{}:0'.format(self.scope, name)


    def feed_dict(self, fd):
        res = {}
        for key in fd:
            res[t(key)] = fd[key]
        return res'''


    def evaluate(self, x, y, session, batch_size=128):
        num = len(x)
        total = 0.
        for offset in range(0, num, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x[offset:end], y[offset:end]
            feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0}
            accuracy = session.run(self.accuracy_op, feed_dict=feed_dict)
            total += accuracy * len(batch_x)
        return total / num


    def predict(self, x, session, rank=1):
        feed_dict = {self.x: [x], self.keep_prob: 1.0}
        prediction = session.run(self.logits, feed_dict=feed_dict)[0]
        _, top = session.run(tf.nn.top_k(tf.constant(np.array(prediction)), k=rank))
        return top[0], list(top[1:])


    def batch_train(self, x, y, session, batch_size=128, keep_prob=0.5):
        num = len(x)
        x, y = shuffle(x, y)
        total = 0.
        for offset in range(0, num, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x[offset:end], y[offset:end]
            feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob}
            _, accuracy = session.run([self.train_op, self.accuracy_op], feed_dict=feed_dict)
            total += accuracy * len(batch_x)
        return total / num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.p', nargs=1, type=str)
    parser.add_argument('--valid', default='valid.p', nargs=1, type=str)
    parser.add_argument('--test', default='test.p', nargs=1, type=str)
    parser.add_argument('--conv1', default=16, nargs=1, type=int)
    parser.add_argument('--conv2', default=32, nargs=1, type=int)
    parser.add_argument('--fc1', default=400, nargs=1, type=int)
    parser.add_argument('--fc2', default=150, nargs=1, type=int)
    parser.add_argument('--name', default='signnet', nargs=1, type=str)
    parser.add_argument('--model-dir', default='model', nargs=1, type=str)
    parser.add_argument('--learning-rate', default=0.001, nargs=1, type=float)
    parser.add_argument('--epochs', default=100, nargs=1, type=int)
    parser.add_argument('--batch-size', default=128, nargs=1, type=int)
    parser.add_argument('--keep-prob', default=0.5, nargs=1, type=float)
    parser.add_argument('--shape', default='32x32x3', nargs=1, type=str)
    parser.add_argument('--num-classes', default=1, nargs=1, type=int)

    args = parser.parse_args()
    args.shape = tuple(map(lambda x: int(x), args.shape.split('x')))

    #x_train, y_train = load_data(args.train)
    x_valid, y_valid = load_data(args.valid)
    #x_test, y_test = load_data(args.test)
    
    net = SignNet(args)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.evaluate(x_valid, y_valid, sess))
        print(net.predict(x_valid[0], sess))
        print(net.batch_train(x_valid, y_valid, sess))

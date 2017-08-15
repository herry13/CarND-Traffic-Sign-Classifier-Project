#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import logging
import pickle
import argparse
from collections import namedtuple
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.python.lib.io import file_io


class SignNet:
    """
    SignNet: a CNN model for Traffic Sign recognition.
    """
    def __init__(self, args=None, session=None, load=False):
        if session is None:
            raise RuntimeError("Parameter 'session' is None!")

        if load:
            self._load_model(session, args.name, args.model_dir)
        elif args is not None:
            self._new_model(session, name=args.name, shape=args.shape,
                    num_classes=args.num_classes, conv1=args.conv1,
                    conv2=args.conv2, fc1=args.fc1, fc2=args.fc2,
                    learning_rate=args.learning_rate)
        else:
            self._new_model(session)
            

    def _load_model(self, session, name, model_dir):
        self.name = name
        meta_file = os.path.join(model_dir, self.name + '.meta')
        
        self.saver = tf.train.import_meta_graph(meta_file)
        self.saver.restore(session, tf.train.latest_checkpoint(os.path.dirname(meta_file)))
        graph = tf.get_default_graph()
        self.scope = tf.variable_scope(self.name)
        tensors = ['x', 'y', 'keep_prob', 'conv1', 'conv2', 'logits', 'steps',
                'cross_entropy', 'loss', 'accuracy']
        for t in tensors:
            name = '{}/{}:0'.format(self.name, t)
            self.__dict__[t] = graph.get_tensor_by_name(name)
            
        operations = ['train_op']
        for op in operations:
            name = '{}/{}'.format(self.name, op)
            self.__dict__[op] = graph.get_operation_by_name(name)


    def _new_model(self, session, name='signnet', shape=(32, 32, 3),
                   num_classes=1, conv1=2, conv2=4, fc1=4, fc2=2,
                   learning_rate=0.001):
        self.name = name
        with tf.variable_scope(self.name) as scope:
            self.scope = scope
            
            # Variables
            x_shape = (None, shape[0], shape[1], shape[2])
            self.x = tf.placeholder(tf.float32, x_shape, name='x')
            self.y = tf.placeholder(tf.int32, (None), name='y')
            one_hot_y = tf.one_hot(self.y, num_classes)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # Preprocess
            preprocess = tf.image.rgb_to_grayscale(self.x)
            #preprocess = (tf.cast(preprocess, tf.float32) - 128.) / 128.
            
            # Convolution layers
            self.conv1 = self._convolution(1, preprocess, conv1, 4, 4, 4)
            self.conv2 = self._convolution(2, self.conv1, conv2, 4, 4, 4)

            # Combine conv1 and conv2
            fc0_1 = flatten(self.conv1)
            fc0_2 = flatten(self.conv2)
            fc0 = tf.concat([fc0_1, fc0_2], 1)

            # Fully-connected layers
            fc1 = self._fully_connected(1, fc0, fc1)
            fc2 = self._fully_connected(2, fc1, fc2)
            self.logits = self._logits(fc2, num_classes)

            # Operations
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=one_hot_y, logits=self.logits,
                    name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy, name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.steps = tf.Variable(0, name='steps', trainable=False)
            self.train_op = optimizer.minimize(self.loss, name='train_op',
                    global_step=self.steps)
            self.correct = tf.equal(tf.argmax(self.logits, 1),
                    tf.argmax(one_hot_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32),
                    name='accuracy')

            # Initialize variables
            session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            
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
        fc = tf.nn.relu6(fc)
        name = 'fc{}'.format(idx)
        fc = tf.nn.dropout(fc, keep_prob=self.keep_prob, name=name)
        return fc
        

    def _convolution(self, idx, x, features, kernelsize, poolsize, poolstridesize):
        # Convolution
        name = 'conv{}_W'.format(idx)
        shape = (kernelsize, kernelsize, int(x.shape[3]), features)
        weights = tf.Variable(self._tf_var(shape), name=name)
        name = 'conv{}_b'.format(idx)
        biases = tf.Variable(tf.zeros(features), name=name)
        conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') \
                    + biases
        # Activation
        conv = tf.nn.relu6(conv)
        # Dropout
        # Max Pooling
        name = 'conv{}'.format(idx)
        conv = tf.nn.avg_pool(conv, ksize=[1, poolsize, poolsize, 1], \
                              strides=[1, poolstridesize, poolstridesize, 1], \
                              padding='VALID', name=name)
        return conv

    
    def evaluate(self, x, y, session, batch_size=128):
        num = len(x)
        total = 0.
        for offset in range(0, num, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x[offset:end], y[offset:end]
            feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0}
            batch_accuracy = session.run(self.accuracy, feed_dict=feed_dict)
            total += batch_accuracy * len(batch_x)
        return float(total / num)


    def predict(self, x, session, rank=1):
        feed_dict = {self.x: [x], self.keep_prob: 1.0}
        prediction = session.run(self.logits, feed_dict=feed_dict)[0]
        _, top = session.run(tf.nn.top_k(tf.constant(np.array(prediction)), k=rank))
        return top


    def batch_train(self, x, y, session, batch_size=128, keep_prob=0.5):
        num = len(x)
        x, y = shuffle(x, y)
        total = 0.
        for offset in range(0, num, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x[offset:end], y[offset:end]
            feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob}
            _, batch_accuracy = session.run([self.train_op, self.accuracy], feed_dict=feed_dict)
            total += batch_accuracy * len(batch_x)
        return float(total / num)


    def save(self, session, directory='', state_only=False, steps=None):
        file_path = os.path.join(directory, self.name)
        meta_file = None
        if not state_only:
            meta_file = self.saver.save(session, file_path, write_state=False)
            tf.logging.info('Saved meta model into {}.'.format(meta_file))
        state_file = self.saver.save(session, file_path,
                global_step=self.steps, write_meta_graph=False)
        tf.logging.info('Saved state model into {}.'.format(state_file))
        return meta_file, state_file


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.p', type=str,
                        help='Training data file (default: train.p)')
    parser.add_argument('--valid', default='valid.p', type=str,
                        help='Validation data file (default: valid.p)')
    parser.add_argument('--test', default='test.p', type=str,
                        help='Test data file (default: test.p)')
    parser.add_argument('--conv1', default=2, type=int)
    parser.add_argument('--conv2', default=4, type=int)
    parser.add_argument('--fc1', default=4, type=int)
    parser.add_argument('--fc2', default=2, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--keep-prob', default=1.0, type=float)
    parser.add_argument('--shape', default='32x32x3', type=str)
    parser.add_argument('--name', default='signnet', type=str)
    parser.add_argument('--model-dir', default='', type=str,
                        help='Model file (default: signnet)')
    parser.add_argument('--predict', default=None, type=str)
    parser.add_argument('--log-file', default=None, type=str,
                        help='Path of log file.')
    parser.add_argument('--save-epochs', default=5, type=int,
                        help='Save the model every N epochs.')

    args, unknown = parser.parse_known_args()
    args.shape = tuple(map(lambda x: int(x), args.shape.split('x')))
    return args


def train(args, retrain=False):
    def load_data(filepath):
        with file_io.FileIO(filepath, 'rb') as f:
            data = pickle.load(f)
            return data['features'], data['labels']

    x_train, y_train = load_data(args.train)
    x_valid, y_valid = load_data(args.valid)
    x_test, y_test = load_data(args.test)
    args.num_classes = len(np.unique(y_train))
    with tf.Session() as sess:
        net = SignNet(args=args, session=sess, load=retrain)
        if not retrain:
            net.save(sess, directory=args.model_dir)
        for i in range(args.epochs):
            train_accuracy = net.batch_train(x_train, y_train, sess,
                    batch_size=args.batch_size, keep_prob=args.keep_prob)
            valid_accuracy = net.evaluate(x_valid, y_valid, sess)
            tf.logging.info('Epoch {} -- Accuracy training={:.5f} validation={:.5f}'.format( \
                    i + 1, train_accuracy, valid_accuracy))
            if (i+1) % args.save_epochs == 0:
                net.save(sess, directory=args.model_dir, state_only=True)

        test_accuracy = net.evaluate(x_test, y_test, sess)
        tf.logging.info('Test accuracy={:.5f}'.format(test_accuracy))

        if args.epochs % args.save_epochs != 0:
            net.save(sess, directory=args.model_dir, state_only=True)


def predict(args):
    with tf.Session() as sess:
        image_file = tf.read_file(args.predict)
        image = sess.run(tf.image.decode_jpeg(image_file))
        net = SignNet(args=args, session=sess, load=True)
        label = net.predict(image, sess)[0]
        tf.logging.info('Label of {} is {}'.format(args.predict, label))


def main():
    args = _parse_arguments()

    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    if args.log_file is not None:
        with file_io.FileIO(args.log_file, 'a') as f:
            logger.addHandler(logging.StreamHandler(f))
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    if args.predict is not None:
        predict(args)
    else:
        try:
            meta_file = os.path.join(args.model_dir, args.name + '.meta')
            with file_io.FileIO(meta_file, 'r'):
                pass
            train(args, retrain=True)
        except:
            train(args)


if __name__ == '__main__':
    main()

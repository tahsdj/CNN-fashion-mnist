#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import keras
from keras.utils import to_categorical # for encoding data to one not form
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import batch as batchHandler

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

def oneHotEncode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded



y_train = oneHotEncode(y_train)
y_test = oneHotEncode(y_test)

# normalize to 0~1  it is important
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def addWeight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def addBias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')




def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# In[43]:


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
batch_size = 800
# print(x_image.shape)  # [n_samples, 28,28,1]


W_conv1 = addWeight([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = addBias([32]) 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32


W_conv2 = addWeight([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = addBias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)   


## fc1 layer ##
W_fc1 = addWeight([7*7*64, 1024])
b_fc1 = addBias([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


## fc2 layer ##
W_fc2 = addWeight([1024, 10])
b_fc2 = addBias([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)


def next_batch(num, data, labels):
    index = np.arange(0 , len(data))
    np.random.shuffle(index)
    idx = idx[:num]
    data_shuffle = [data[i] for i in index]
    labels_shuffle = [labels[i] for i in index]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# create a class for dealing with our data for training process
class Data():
    def __init__(self,samples,labels):
        self.samples = samples
        self.labels = labels
        self.pointer = 0
    def next_batch(self,num):
        if self.pointer+num <= len(self.samples):
            batch_samples = self.samples[self.pointer:self.pointer+num]
            batch_labels = self.labels[self.pointer:self.pointer+num]
            self.pointer += num
            return batch_samples, batch_labels
        else:
            new_pointer = self.pointer+num - len(self.samples)
            batch_samples = self.samples[self.pointer:-1] + self.samples[:new_pointer]
            batch_labels = self.labels[self.pointer:-1] + self.labels[:new_pointer]
            self.pointer = new_pointer
            return batch_samples, batch_labels

training_data = Data(x_train,y_train) #put our trainging data into this class

# learning_curve_data = []


# In[60]:


for i in range(20000):
    batch_xs, batch_ys = training_data.next_batch(batch_size)
    batch_xs = batch_xs.reshape((batch_size,784))
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        acc = compute_accuracy(x_test.reshape(len(x_test),784), y_test)
        print('accuracy: ', acc)
        # learning_curve_data.append(acc)   


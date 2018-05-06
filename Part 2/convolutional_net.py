import random
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
import scipy.ndimage

###############################################################################
# DATA

inputs = []
outputs = []

classes = ['pigs', 'cages']

'''fills inputs with np.arrays of images and fills outputs with corresponding 
one-hot vectors'''
for index, field in enumerate(classes):
    print('reading {} files (Index: {})'.format(field, index))
    path = os.path.dirname(__file__) + '/' + field
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        input_array = scipy.ndimage.imread(
            filename, flatten=True, mode=None)
        inputs.append(input_array)
        output_vec = np.array([0, 0])
        output_vec[index] = 1
        outputs.append(output_vec)


def get_batch(batch_size):
    '''returns a number of randomly chosen input-output pairs'''
    batch_inputs = []
    batch_outputs = []
    random_indices = random.sample(range(1, len(inputs)), batch_size)
    for i in random_indices:
        batch_inputs.append(inputs[i])
        batch_outputs.append(outputs[i])
    x_ = np.array(batch_inputs)
    y_ = np.array(batch_outputs)
    return tuple((x_, y_))

###############################################################################
# PARAMETERS

'''Network architecture is as follows:
Input - Conv - MaxPool - Conv - MaxPool - FullyCnctd - Output
'''
# Layer Sizes
input_height = 148
input_width = 200
conv1_size = 10
conv1_depth = 16
conv2_size = 10
conv2_depth = 32
fullyconnected_size = 256
output_size = 2 
# Training Hyperparameters
batch_size = 5
learning_rate = 0.0001
dropout_rate = 0.4

###############################################################################
# NETWORK

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, input_height, input_width])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# functions to create new tf layers
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, input_height, input_width, 1])

# Convolution + MaxPool Layer 1
W_conv1 = weight_variable([conv1_size, conv1_size, 1, conv1_depth])
b_conv1 = bias_variable([conv1_depth])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolution + MaxPool Layer 2
W_conv2 = weight_variable([conv2_size, conv2_size, conv1_depth, conv2_depth])
b_conv2 = bias_variable([conv2_depth])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected Layer
maxpool2_size = int(input_height * input_width * conv2_depth / 16)
W_fc1 = weight_variable([maxpool2_size, fullyconnected_size])
b_fc1 = bias_variable([fullyconnected_size])
h_pool2_flat = tf.reshape(h_pool2, [-1, maxpool2_size])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output Layer
W_fc2 = weight_variable([fullyconnected_size, output_size])
b_fc2 = bias_variable([output_size])
output_y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define Cross Entropy Cost Function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_y))

# Set Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Evaluates training accuracy of a batch
correct_prediction = tf.equal(tf.argmax(output_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

###############################################################################
# TRAINING

for i in range(1000):
    batch = get_batch(batch_size)

    if i % 20 == 0:
        # show testing accuracy
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # update weights
    train_step.run(feed_dict={x: batch[0], y_: batch[1], 
    keep_prob: dropout_rate})
    print('step {} complete'.format(i))

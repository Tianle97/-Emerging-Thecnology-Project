# Tianle Shu
# Import the package
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# For Ignore the warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# get the mnist dataset save it in the current folder and named "data" 
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. 
# In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.
# For example, 3 would be [0,0,0,1,0,0,0,0,0,0]
mnist = input_data.read_data_sets('data',one_hot=True)


# set batch_size, every time put 100 images for training
batch_size = 100
# calculate number of batches from data set
n_batch = mnist.train.num_examples // batch_size

# Define two placehoder
x = tf.placeholder(tf.float32,[None,784]) 
y = tf.placeholder(tf.float32,[None,10])

# Define a nural network              
# tf.zeros -> initial value is 0
# W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it 
# to produce 10-dimensional vectors of evidence for the difference classes. 
W = tf.Variable(tf.zeros([784,10])) 
b = tf.Variable(tf.zeros([10]))     # b:bisa, has a shape of [10] so we can add it to the output.
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# Calculate cost(loss), and minimize loss
# Use  quadratic cost method --fomular (y-mx-b)^2  ---  suit for linear
#     loss = tf.reduce_mean(tf.square(y-prediction))  --- suit for s-shape
# use cross_entropy method: tf.nn.softmax_cross_entropy_with_logits method
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# use AdamOptimizer to train
# More details about AdamOptimizer: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)   # 1e-2 = 10^-2 = 0.02

# initial glabal variables
init = tf.global_variables_initializer()

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) # correct return true, otherwise return false
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # true->1.0   false->0

sum_acc = 0
loop = 300
# create loop to train
with tf.Session() as sess:
    sess.run(init)
    for eposh in range(loop):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,{x:batch_xs, y:batch_ys})

# calculate accuracy
        acc = sess.run(accuracy,{x:mnist.test.images, y:mnist.test.labels})
        if(eposh%20 ==0):
            print("Iter " + str(eposh) + ",Testing Accuracy " + str(acc))
        sum_acc += acc
print("Accuracy Rate" + str((sum_acc / loop) * 100)+"%")

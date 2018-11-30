# Tianle Shu
# Import the package
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# For Ignore the warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# set batch_size, every time put 100 images for training
batch_size = 100
# calculate number of batches from data set
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784]) 
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10])) 
b = tf.Variable(tf.zeros([10]))     # b:bisa, has a shape of [10] so we can add it to the output.
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)   # 1e-2 = 10^-2 = 0.02

# initial glabal variables
init = tf.global_variables_initializer()

# calculate accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) # correct return true, otherwise return false
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # true->1.0   false->0

# create loop to train
with tf.Session() as sess:
    sess.run(init)
    for eposh in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,{x:batch_xs, y:batch_ys})

# calculate accuracy
        acc = sess.run(accuracy,{x:mnist.test.images, y:mnist.test.labels})
        print("Iter " + str(eposh) + ",Testing Accuracy " + str(acc))

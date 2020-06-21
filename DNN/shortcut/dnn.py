
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
learning_rate = 0.01
training_epochs = 15
batch_size = 100

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)
L0 = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

W1 = tf.get_variable("W1", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W6 = tf.get_variable("W6", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W7 = tf.get_variable("W7", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W8 = tf.get_variable("W8", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W9 = tf.get_variable("W9", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W10 = tf.get_variable("W10", shape = [784, 10], initializer = tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.random_normal([784]))
B2 = tf.Variable(tf.random_normal([784]))
B3 = tf.Variable(tf.random_normal([784]))
B4 = tf.Variable(tf.random_normal([784]))
B5 = tf.Variable(tf.random_normal([784]))
B6 = tf.Variable(tf.random_normal([784]))
B7 = tf.Variable(tf.random_normal([784]))
B8 = tf.Variable(tf.random_normal([784]))
B9 = tf.Variable(tf.random_normal([784]))
B10 = tf.Variable(tf.random_normal([10]))
# B4 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(L0, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), B5))
L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), B6))
L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), B7))
L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), B8))
L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), B9))
hypothesis = tf.add(tf.matmul(L9, W10), B10)


val = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis)
cost = tf.reduce_mean(val)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for step in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={L0: batch_xs, Y: batch_ys})

            avg_cost += sess.run(cost, feed_dict={L0: batch_xs, Y: batch_ys})/total_batch

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy=", accuracy.eval({L0: mnist.test.images, Y: mnist.test.labels}))

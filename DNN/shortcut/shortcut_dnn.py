
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
# W3 = tf.get_variable("W3", shape = [784, 784], initializer = tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W4", shape = [784, 10], initializer = tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.random_normal([784]))
B2 = tf.Variable(tf.random_normal([784]))
B3 = tf.Variable(tf.random_normal([10]))
# B4 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(L0, W1), B1))
L1_sc = tf.add(L0, L1)
L2 = tf.nn.relu(tf.add(tf.matmul(L1_sc, W2), B2))
L2_sc = tf.add(L1_sc, L2)
# L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
hypothesis = tf.add(tf.matmul(L2_sc, W3), B3)


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

import subprocess


def fileMaker(gene):
    lr = gene[0]
    initW = gene[1]
    optim = gene[2]
    actF = gene[3]
    layer = gene[4]
    # fitness = gene[5]

    f = open("created_dnn.py", 'w')

    f.write("\n")
    f.write("import sys\n")
    f.write("import tensorflow as tf\n")
    f.write("from tensorflow.examples.tutorials.mnist import input_data\n")
    f.write("learning_rate = " + str(lr) + "\n")
    f.write("training_epochs = 15\n")
    f.write("batch_size = 100\n\n")

    f.write("mnist = input_data.read_data_sets(\"./MNIST_DATA\", one_hot = True)\n")
    f.write("L0 = tf.placeholder('float', [None, 784])\n")
    f.write("Y = tf.placeholder('float', [None, 10])\n\n")

    node = [784]
    for i in range(1, layer+3):
        if(i == (layer+2)):
            node.append(10)
        else:
            node.append(int(784/(layer+1)*(layer+2-i)))

    if(initW == 'he'):
        for i in range(1, layer+2):
            f.write("W" + str(i) + " = tf.get_variable(\"W" + str(i) + "\", shape = [" + str(node[i]) + ", " + str(node[i+1]) + "], initializer = tf.initializers.he_normal(seed=None))\n")
    elif(initW == 'zeros'):
        for i in range(1, layer+2):
            f.write("W" + str(i) + " = tf.Variable(tf.zeros([" + str(node[i]) + ", " + str(node[i+1]) + "]))\n")
    elif(initW == 'xavier'):
        for i in range(1, layer+2):
            f.write("W" + str(i) + " = tf.get_variable(\"W" + str(i) + "\", shape = [" + str(node[i]) + ", " + str(node[i+1]) + "], initializer = tf.contrib.layers.xavier_initializer())\n")
    elif(initW == 'random'):
        for i in range(1, layer+2):
            f.write("W" + str(i) + " = tf.Variable(tf.random_normal([" + str(node[i]) + ", " + str(node[i+1]) + "]))\n")
    f.write("\n")

    for i in range(1, layer+2):
        f.write("B" + str(i) + " = tf.Variable(tf.random_normal([" + str(node[i+1]) + "]))\n")
    f.write("\n")

    if(actF == 'relu'):
        for i in range(1, layer+1):
            f.write("L" + str(i) + " = tf.nn.relu(tf.add(tf.matmul(L" + str(i-1) + ", W" + str(i) + "), B" + str(i) + "))\n")
    elif(actF == 'sigmoid'):
        for i in range(1, layer+1):
            f.write("L" + str(i) + " = tf.div(1., 1. + tf.exp(-(tf.add(tf.matmul(L" + str(i-1) + ", W" + str(i) + "), B" + str(i) + "))))\n")
    elif(actF == 'tanh'):
        for i in range(1, layer+1):
            f.write("L" + str(i) + " = tf.math.tanh(tf.add(tf.matmul(L" + str(i-1) + ", W" + str(i) + "), B" + str(i) + "))\n")

    f.write("hypothesis = tf.add(tf.matmul(L" + str(layer) + ", W" + str(layer+1) + "), B" + str(layer+1) + ")\n\n")

    f.write("val = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = hypothesis)\n")
    f.write("cost = tf.reduce_mean(val)\n")

    if(optim == 'Adam'):
        f.write("optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)\n\n")
    elif(optim == 'Adagrad'):
        f.write("optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)\n\n")
    elif(optim == 'SGD'):
        f.write("optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n\n")

    f.write("init = tf.initialize_all_variables()\n\n")
    f.write("with tf.Session() as sess:\n")
    f.write("    sess.run(init)\n\n")
    f.write("    for epoch in range(training_epochs):\n")
    f.write("        avg_cost = 0.\n")
    f.write("        total_batch = int(mnist.train.num_examples/batch_size)\n\n")
    f.write("        for step in range(total_batch):\n")
    f.write("            batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n\n")
    f.write("            sess.run(optimizer, feed_dict = {L0: batch_xs, Y:batch_ys})\n\n")
    f.write("            avg_cost += sess.run(cost, feed_dict = {L0:batch_xs, Y:batch_ys})/total_batch\n\n")
    f.write("        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))\n\n")
    f.write("    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \"float\"))\n\n")
    f.write("    print(\"Accuracy=\", accuracy.eval({L0:mnist.test.images, Y:mnist.test.labels}), \"genetic\")")

    f.close()

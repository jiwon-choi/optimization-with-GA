# -*- coding: utf-8 -*-

import subprocess

def fileMaker(gene, index = None, parent = None):

    fitness = gene[0]
    lr = gene[1]
    initW = gene[2]
    optim = gene[3]
    actF = gene[4]
    kernel_size = gene[5]
    conv_layer = gene[6][0]
    n_conv = gene[6][1]
    fc_layer = gene[7]
    drop_out = gene[8]

    f = open("created_cnn.py", 'w')

    # import
    f.write("\n")
    f.write("import tensorflow as tf\n")
    f.write("from tensorflow import keras\n")
    f.write("from tensorflow.keras import layers\n")
    f.write("from tensorflow.keras import datasets\n")
    f.write("import copy\n")
    f.write("import numpy as np\n\n")

    # gene
    f.write("lr = " + str(lr) + "\n")
    f.write("initW = '" + str(initW) + "'\n")
    if optim == 'Adam':
        f.write("opt = keras.optimizers.Adam(lr =lr, beta_1=0.9, beta_2=0.999, amsgrad=False)\n")
    elif optim == 'Adagrad':
        f.write("opt = keras.optimizers.Adagrad(learning_rate=lr)\n")
    elif optim == 'SGD':
        f.write("opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)\n")
    elif optim == 'Adadelta':
        f.write("opt = keras.optimizers.Adadelta(learning_rate=lr, rho=0.95)\n")
    f.write("actF = '" + str(actF) + "'\n")
    f.write("ks = " + str(kernel_size) + "\n")
    f.write("conv_layer = " + str(conv_layer) + "\n")
    f.write("fc_layer = " + str(fc_layer) + "\n")
    f.write("drop_out = " + str(drop_out) + "\n")
    f.write("n_conv = " + str(n_conv) + "\n\n")

    f.write("img_rows = 28\n")
    f.write("img_cols = 28\n\n")

    f.write("(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()\n\n")

    f.write("input_shape = (img_rows, img_cols, 1)\n")
    f.write("x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n")
    f.write("x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n")
    f.write("x_train = x_train.astype('float32') / 255.\n")
    f.write("x_test = x_test.astype('float32') / 255.\n\n")

    f.write("batch_size = 128\n")
    f.write("num_classes = 10\n")
    f.write("epochs = 10\n\n")

    f.write("y_train = keras.utils.to_categorical(y_train, num_classes)\n")
    f.write("y_test = keras.utils.to_categorical(y_test, num_classes)\n\n")

    # !
    f.write("inputs = keras.Input(shape = input_shape, name = 'input')\n")
    # f.write("output = copy.deepcopy(inputs)\n")

    for _ in range(conv_layer):
        f.write("identity = layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same', activation = actF)(inputs)\n")
        f.write("output = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')(identity)\n")
        if n_conv >= 2:
            for _ in range(n_conv-1):
                f.write("output = layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same')(output)\n")
        f.write("output = layers.BatchNormalization()(output)\n")
        f.write("output = layers.MaxPooling2D(pool_size = [ks, ks], padding = 'same', strides = 1)(output)\n")
        f.write("dropout = layers.Dropout(rate=drop_out)(output)\n")
        f.write("output = layers.Activation(actF)(dropout)\n")
        f.write("output = layers.Add()([output, identity])\n\n")

    f.write("output = layers.GlobalAveragePooling2D()(output)\n")
    # Dense 수정
    f.write("output = layers.Dense(1000, activation = actF)(output)\n")
    # f.write("output = layers.BatchNormalization()(output)\n")
    f.write("dropout = layers.Dropout(rate=drop_out)(output)\n")
    f.write("output = layers.Dense(10, activation = 'softmax')(dropout)\n\n")

    f.write("model = keras.Model(inputs = inputs, outputs = output)\n")
    f.write("model.summary()\n\n")

    f.write("model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])\n")

    if parent != None:
        f.write("pmodel = keras.models.load_model(\"model_" + str(parent) + ".h5\")\n")
        fp = open("./saved/chromosome_" + str(parent) + ".txt", "r")
        p_kernel_size = int(fp.readline())
        p_conv_layer = int(fp.readline())
        p_n_conv = int(fp.readline())
        fp.close()
        for i in range(min(conv_layer, p_conv_layer)):
            for j in range(min(n_conv+1, p_n_conv+1)):
                if kernel_size < p_kernel_size:
                    f.write("k = pmodel.layers[" + str(1 + j + i*5 + i*(p_n_conv+1)) + "].get_weights()[0]\n")
                    f.write("k = k[:" + str(kernel_size) + ", :" + str(kernel_size) + ", :, :" +  "]\n")
                    f.write("b = pmodel.layers[" + str(1 + j + i*5 + i*(p_n_conv+1)) + "].get_weights()[1]\n")
                    f.write("w = [k,b]\n")
                elif kernel_size > p_kernel_size:
                    padding = str(kernel_size-p_kernel_size)
                    f.write("k = pmodel.layers[" + str(1 + j + i*5 + i*(p_n_conv+1)) + "].get_weights()[0]\n")
                    f.write("k = np.pad(k, (("+padding+","+padding+"),("+padding+","+padding+"),(0,0),(0,0)), \'constant\', constant_values=0)\n")
                    f.write("b = pmodel.layers[" + str(1 + j + i*5 + i*(p_n_conv+1)) + "].get_weights()[1]\n")
                    f.write("w = [k,b]\n")
                else:
                    f.write("pmodel.layers[" + str(1 + j + i*5 + i*(p_n_conv+1)) + "].get_weights()\n")
                f.write("model.layers[" + str(1 + j + i*5 + i*(n_conv+1)) + "].set_weights(w)\n")
        fc = conv_layer*5 + conv_layer*(n_conv+1)
        p_fc = p_conv_layer*5 + p_conv_layer*(p_n_conv+1)
        f.write("model.layers[" + str(fc+2) + "].set_weights(pmodel.layers["+ str(p_fc+2) +"].get_weights())\n")
        f.write("model.layers[" + str(fc+4) + "].set_weights(pmodel.layers["+ str(p_fc+4) +"].get_weights())\n")

    f.write("hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))\n\n")

    f.write("score = model.evaluate(x_test, y_test, verbose=0)\n")
    f.write("print(\"Accuracy=\", score[1], \"genetic\")\n")
    
    if index == None: print("wrong index")
    f.write("model.save(\'./saved/model_" + str(index) + ".h5\')\n")

    f.close()

import subprocess

def fileMaker(gene):
    lr = gene[0]
    initW = gene[1]
    optim = gene[2]
    actF = gene[3]
    conv_layer = 3
    fcn_layer = 3
    f = open("created_cnn.py", 'w')

    # Import Part

    f.write("\n")
    f.write("import sys\n")
    f.write("import keras\n")
    f.write("import numpy as np\n\n")

    f.write("from keras import optimizers\n")
    f.write("from keras.models import Sequential\n")
    f.write("from keras.layers import Dense, Dropout, Flatten\n")
    f.write("from keras.layers.convolutional import Conv2D, MaxPooling2D\n")
    f.write("from keras.utils import multi_gpu_model\n\n")

    # Gene

    f.write("lr = " + str(lr) + "\n")
    f.write("actF = " + str(actF) + "\n")
    if optim == 'Adam':
        f.write("opt = keras.optimizers.Adam(learning_rate =lr, beta_1=0.9, beta_2=0.999, amsgrad=False)\n")
    elif optim == 'Adagrad':
        f.write("opt = keras.optimizers.Adagrad(learning_rate=lr)\n")
    elif optim == 'SGD':
        f.write("opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)\n")
    elif optim == 'Adadelta':
        f.write("opt = keras.optimizers.Adadelta(learning_rate=lr, rho=0.95)\n")

    #

    f.write("img_rows = 28\n")
    f.write("img_cols = 28\n\n")

    f.write("(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()\n")

    f.write("input_shape = (img_rows, img_cols, 1)\n")
    f.write("x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n")
    f.write("x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n")
    f.write("x_train = x_train.astype('float32') / 255.\n")
    f.write("x_test = x_test.astype('float32') / 255.\n\n")

    f.write("batch_size = 128\n")
    f.write("num_classes = 10\n")
    f.write("epochs = 10\n\n")

    f.write("y_train = keras.utils.to_categorical(y_train, num_classes)\n")
    f.write("y_test = keras.utils.to_categorical(y_test, num_classes)\n")

    # Structure Part
    f.write("model = Sequential()\n")
    f.write("model.add(Conv2D(32,kernel_size=(ks, ks), padding='same', activation = 'relu',input_shape=input_shape))\n")
    f.write("model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n")
    for i in range(0, conv_layer-1):
        f.write("model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))\n")
        f.write("model.add(MaxPooling2D(pool_size=(2, 2)))\n")
    f.write("model.add(Dropout(0.25))\n")
    f.write("model.add(Flatten())\n")
    for i in range(0, fcn_layer-2):
        f.write("model.add(Dense(1000, activation='relu'))\n")
        f.write("Dropout(0.5)\n")
    f.write("model.add(Dense(num_classes, activation='softmax'))\n")
    f.write("model.summary()\n\n")

    '''
    f.write("model = Sequential()\n")
    f.write("model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',activation= 'relu',input_shape=input_shape))\n")
    f.write("model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n")
    f.write("model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))\n")
    f.write("model.add(MaxPooling2D(pool_size=(2, 2)))\n")
    f.write("model.add(Dropout(0.25))\n")
    f.write("model.add(Flatten())\n")
    f.write("model.add(Dense(1000, activation='relu'))\n")
    f.write("model.add(Dropout(0.5))\n")
    f.write("model.add(Dense(num_classes, activation='softmax'))\n")
    f.write("model.summary()\n\n")
    '''
    f.write("model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])\n")
    f.write("hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))\n")

    f.write("score = model.evaluate(x_test, y_test, verbose=0)\n")
    f.write("print(\"Accuracy=\", score[1], \"genetic\")")

    f.close()

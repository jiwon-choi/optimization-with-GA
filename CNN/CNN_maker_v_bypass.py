import subprocess

def fileMaker(gene):
    # lr = gene[0]

    f = open("created_CNN.py", 'w')
    
    f.write("\n")
    f.write("import tensorflow as tf\n")
    f.write("from tensorflow import keras\n")
    f.write("from tensorflow.keras import layers\n")
    f.write("from tensorflow.keras import datasets\n\n")

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
    


    f.write("model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])\n")
    f.write("hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))\n")

    f.write("score = model.evaluate(x_test, y_test, verbose=0)\n")
    f.write("print(\"Accuracy=\", score[1], \"genetic\")")

    f.close()

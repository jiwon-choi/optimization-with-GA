
import sys
import keras
import numpy as np

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model

kernel_init ='he_uniform'
lr = 0.7868
actF = 'tanh'
ks =1
opt = keras.optimizers.Adagrad(learning_rate=lr)
img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

batch_size = 128
num_classes = 10
epochs = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Conv2D(32,kernel_size=(ks, ks), kernel_initializer = kernel_init, padding='same', activation = actF,input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(64, (2, 2), activation=actF, padding='same', kernel_initializer = kernel_init))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, (2, 2), activation=actF, padding='same', kernel_initializer = kernel_init))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy=", score[1], "genetic")
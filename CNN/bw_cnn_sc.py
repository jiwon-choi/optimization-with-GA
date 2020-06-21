import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten,BatchNormalization
#from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets

print('Python version : ', sys.version)
print('Keras version : ', keras.__version__)

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 30
filename = 'checkpoint.h5'.format(epochs, batch_size)
'''
early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=15,verbose=1)                           #얼리스타핑
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')           #체크포인트
'''

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = keras.Input(shape=input_shape, name='input' )
# 1
identity = layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation='relu')(inputs)
output = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')(identity)
output = layers.BatchNormalization()(output)
dropout = layers.Dropout(rate=0.25)(output)
output = layers.Activation('relu')(dropout)
output = layers.Add()([output, identity])
# 2
identity = layers.Conv2D(filters=64, kernel_size=[3,3], padding='same', activation='relu')(output)  #수정필요
output = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')(identity)
output = layers.BatchNormalization()(output)
output = layers.MaxPooling2D(pool_size=[3, 3], padding='same', strides=1)(output)
dropout = layers.Dropout(rate=0.25)(output)
output = layers.Activation('relu')(dropout)
output = layers.Add()([output, identity])
# 3
identity = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same')(output)
output = layers.BatchNormalization()(identity)
output = layers.MaxPooling2D(pool_size=[3, 3], padding='same', strides=1)(output)
dropout = layers.Dropout(rate=0.25)(output)
output = layers.Activation('relu')(dropout)
output = layers.Add()([output, identity])

# 4
output = layers.GlobalAveragePooling2D()(output)
output = layers.Dense(1000, activation='relu')(output)
# output = layers.BatchNormalization()(output)
dropout = layers.Dropout(rate=0.25)(output)
output = layers.Dense(10, activation='softmax')(dropout)

model = keras.Model(inputs=inputs, outputs=output)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',  score[0])
print('Test accuracy:', score[1])
model.save('MNIST_CNN_model.h5')

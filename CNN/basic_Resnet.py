import tensorflow as tf
from tensorflow import keras
from tensorflow.Keras import layers
from tensorflow.Keras import datasets

n = 5
inputs = keras.Input(shape = (32, 32, 3), name = 'input')
identity = layers.Conv2D(filters = 16, kernel_size = [7, 7], padding = 'Same', activation = 'relu')(inputs)

# block1
for _ in range(n):
    output = layers.Conv2D(filter = 16, kernel_size = [3, 3], padding = 'Same')(identity)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filter = 16, kernel_size = [3, 3], padding = 'Same')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Add()([output, identity])
    identity = layers.Activation('relu')(output)

identity = layers.MaxPooling2D(pool_size = [3, 3], padding = 'same', strides = 2)(identity)

# block2
identity = layers.ZeroPadding2D([0, 8], 'channels_first')(identity)
for _ in range(n):
    output = layers.Conv2D(filter = 32, kernel_size = [3, 3], padding = 'Same')(identity)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filter = 32, kernel_size = [3, 3], padding = 'Same')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Add()([output, identity])
    identity = layers.Activation('relu')(output)

identity = layers.MaxPooling2D(pool_size = [3, 3], padding = 'same', strides = 2)(identity)

# block3
identity = layers.ZeroPadding2D([0, 16], 'channels_first')(identity)
for _ in range(n):
    output = layers.Conv2D(filter = 64, kernel_size = [3, 3], padding = 'Same')(identity)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)
    output = layers.Conv2D(filter = 64, kernel_size = [3, 3], padding = 'Same')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Add()([output, identity])
    identity = layers.Activation('relu')(output)

identity = layers.MaxPooling2D(pool_size = [3, 3], padding = 'same', strides = 2)(identity)

output = layers.GlobalAveragePooling2D()(identity)
output = layers.Dense(128, activation = 'relu')(output)
output = layers.Dense(128, activation = 'relu')(output)
output = layers.Dense(10, activation = 'softmax')(output)

model = keras.Model( inputs = inputs, outputs = output, name = 'resnet' )
model.summary()
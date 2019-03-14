from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

MODEL_OUTPUT_DIRECTORY = 'model_weights'

JSON_OUTPUT = 'model.json'
H5_OUTPUT = 'model.h5'

model = Sequential([

    Reshape((160, 320, 1), input_shape=(160, 320)),

    Conv2D(48, 8, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 77x157
    Conv2D(36, 5, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 37x77
    Conv2D(48, 5, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 17x37
    Conv2D(64, 3, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 8x18
    Conv2D(64, 2, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 4x9
    Flatten(),

    Dense(1024),
    Dropout(0.5),
    Activation('relu'),

    Dense(512),
    Dropout(0.5),
    Activation('relu'),

    Dense(256),
    Activation('relu'),

    Dense(128),
    Activation('relu'),

    Dense(32),
    Activation('tanh'),

    Dense(1)
])

model.load_weights(MODEL_OUTPUT_DIRECTORY + '/' + '0.0020.hdf5')

with open(JSON_OUTPUT, 'w') as file:
    file.write(model.to_json())
    file.close()

model.save_weights(H5_OUTPUT)


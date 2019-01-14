from keras.models import Sequential

import properties as properties
import random as rand
import numpy as np
import cv2
import logging
import csv
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Reshape
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback as KerasCallback

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

IMAGE_DATA_DIRECTORY = 'C:/Users/Paperspace/project/torcs-1.3.7/runtimed/E_Track_6_Images/'
INPUT_IMAGE_FORMAT = '.bmp'

STEERING_ANGLE_COLUMN = 'steer'
IMAGE_FILE_COLUMN = 'count'

INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 480
CROPPED_IMAGE_HEIGHT = 320

STRAIGHT_DRIVING_INCLUSION_RATE = 1/(1-properties.STRAIGHT_DRIVING_EXCLUSION_FACTOR)


"""
Retrieve sensor data from torcs-1.3.7;
Discards most samples with minimal steering input according to STEERING_BOUNDARY and STRAIGHT_DRIVING_INCLUSION_RATE 
Returns:
    sensor_data_tuples: a list of tuples each containing pairs of:
        1) steering angle
        2) corresponding image path
"""
def load_sensor_data():

    logging.info(' Retrieving sensor data from ' + properties.SENSOR_DATA_DIRECTORY + properties.SENSOR_CSV_FILE)
    steering_data_list = []
    image_file_list = []

    with open(properties.SENSOR_DATA_DIRECTORY + properties.SENSOR_CSV_FILE) as sensor_data_csv:
        sensor_data_reader = csv.DictReader(sensor_data_csv, dialect="excel")

        straight_driving_image_count = 0

        for row in sensor_data_reader:

            if row[STEERING_ANGLE_COLUMN] == '':
                continue

            try:
                steering_angle = float(row[STEERING_ANGLE_COLUMN])

            except ValueError:
                logging.WARNING('Invalid steering angle entry: ' + row[STEERING_ANGLE_COLUMN] + ' for image ' +
                                row[IMAGE_FILE_COLUMN] + '; Value excluded from training data.')

            if abs(steering_angle) < properties.STEERING_BOUNDARY:

                straight_driving_image_count += 1

                if straight_driving_image_count == STRAIGHT_DRIVING_INCLUSION_RATE:

                    steering_data_list.append(steering_angle)
                    image_file_list.append(row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT)
                    straight_driving_image_count = 0

            else:

                steering_data_list.append(steering_angle)
                image_file_list.append(row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT)

    return steering_data_list, image_file_list


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
"""
Retrieve image data corresponding to a list of image files located in IMAGE_DATA_DIRECTORY
Args:
    image_paths: a list of image file names
Returns:
    a Numpy array of image data corresponding to each image
"""
def load_images(steering_angles, image_files):

    logging.info(' Loading images from ' + IMAGE_DATA_DIRECTORY)
    y_start = INPUT_IMAGE_HEIGHT - CROPPED_IMAGE_HEIGHT
    x_start = 0
    image_data_set = []
    vertices = np.array([[0, 385], [0, 275], [450, 250], [640, 275], [640, 385]])
    count = -1

    for current_image in image_files:

        count += 1

        if current_image is None:
            del steering_angles[count]
            continue

        og_image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.COLOR_BGR2RGB)
        blurred_image = cv2.GaussianBlur(og_image, (3, 3), 0)
        edge_image = cv2.Canny(blurred_image, 50, 150)
        roi_image = roi(edge_image, [vertices])
        cropped_image = roi_image[250:410, x_start:INPUT_IMAGE_WIDTH]
        downsize_image = cv2.resize(cropped_image, (0, 0), fx=0.5, fy=1)

        #image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.figure(1)
        #plt.imshow(roi_image)
        #plt.show()
        #cv2.imshow('original', image)
        #cv2.waitKey(0)

        #cv2.imshow('edges', image)
        #cv2.waitKey(0)
        #vertices = np.array([[0, 385], [0, 275], [450, 250], [640, 275], [640, 385]])

        #cv2.imshow('roi', image)
        #cv2.waitKey(0)

        #cv2.imshow('Input Image', downsize_image)
        #cv2.waitKey(0)

        image_data_set.append(downsize_image)

    return steering_angles, np.array(image_data_set)


steering_angles, image_paths = load_sensor_data()
steering_angles, image_data = load_images(steering_angles, image_paths)

model = Sequential([

    Reshape((160, 320, 1), input_shape=(160, 320)),

    Convolution2D(24, 8, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 77x157
    Convolution2D(36, 5, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 37x77
    Convolution2D(48, 5, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 17x37
    Convolution2D(64, 3, padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Activation('relu'),

    # 8x18
    Convolution2D(64, 2, padding='valid'),
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

optimizer = Adam(lr=1e-4)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[]
)

epochs = 10


class SaveModel(KerasCallback):

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if epoch > 9:
            with open('model-' + str(epoch) + '.json', 'w') as file:
                file.write(model.to_json())
                file.close()

            model.save_weights('model-' + str(epoch) + '.h5')


save_model = SaveModel()

model.fit(
    x=np.array(image_data),
    y=np.array(steering_angles),
    epochs=epochs,
    validation_split=0.33,
    callbacks=[save_model]
)
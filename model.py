from keras.models import Sequential

import numpy as np
import cv2
import logging
import csv
from keras.layers import Dense, Activation, Reshape
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback as KerasCallback

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

SENSOR_DATA_DIRECTORY = 'C:/Users/Derek/Source/Repos/torcs-1.3.7/runtimed/'
SENSOR_CSV_FILE = 'sensors.csv'

IMAGE_DATA_DIRECTORY = 'C:/Users/Derek/Source/Repos/torcs-1.3.7/runtimed/images/'
INPUT_IMAGE_FORMAT = '.bmp'

STEERING_ANGLE_COLUMN = 'steer'
IMAGE_FILE_COLUMN = 'count'

INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 480
CROPPED_IMAGE_HEIGHT = 320

STEERING_BOUNDARY = 0.00005
STRAIGHT_DRIVING_INCLUSION_RATE = 10

"""
Retrieve sensor data from torcs-1.3.7;
Discards most samples with minimal steering input according to STEERING_BOUNDARY and STRAIGHT_DRIVING_INCLUSION_RATE 
Returns:
    sensor_data_tuples: a list of tuples each containing pairs of:
        1) steering angle
        2) corresponding image path
"""
def load_sensor_data():

    logging.info(' Retrieving sensor data from ' + SENSOR_DATA_DIRECTORY + SENSOR_CSV_FILE)
    sensor_data_tuples = []

    with open(SENSOR_DATA_DIRECTORY + SENSOR_CSV_FILE) as sensor_data_csv:
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

            if abs(steering_angle) < STEERING_BOUNDARY:

                straight_driving_image_count += 1

                if straight_driving_image_count == STRAIGHT_DRIVING_INCLUSION_RATE:

                    sensor_data_tuples.append((steering_angle, row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT))
                    straight_driving_image_count = 0

            else:

                sensor_data_tuples.append((steering_angle, row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT))

    return sensor_data_tuples


"""
Retrieve image data corresponding to a list of image files located in IMAGE_DATA_DIRECTORY
Args:
    image_paths: a list of image file names
Returns:
    a Numpy array of image data corresponding to each image
"""
def load_images(image_files):

    logging.info(' Loading images... ')
    y_start = INPUT_IMAGE_HEIGHT - CROPPED_IMAGE_HEIGHT
    x_start = 0
    image_data_set = []

    for current_image in image_files:

            image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.IMREAD_GRAYSCALE)
            cropped_image = image[y_start:INPUT_IMAGE_HEIGHT, x_start:INPUT_IMAGE_WIDTH]
            image_data_set.append(cropped_image)

    return np.array(image_data_set)


def partition_training_data(training_data, test_proportion):

    logging.info(' Partitioning training data... ')
    training_set_count = int((1.0-test_proportion)*len(training_data[0]))
    train_set = [training_data[0:training_set_count] for list in training_data]
    valid_set = [training_data[training_set_count:] for list in training_data]
    return train_set, valid_set


"""
Shuffle image and steering angle lists (indexing)
Args:
    training_data: list of lists of equal length
Returns:
    list of shuffled lists
"""
def shuffle_data(training_data):

    permuted_sequence = np.random.permutation(len(training_data[0]))
    return [training_data[permuted_sequence] for list in training_data]


sensor_data = load_sensor_data()
steering_angles, image_paths = zip(*sensor_data)

steering_angles = np.asarray(steering_angles)
image_data = load_images(image_paths)

#TODO: Pooling/Dropout Layers?
model = Sequential([
        Reshape((CROPPED_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1), input_shape=(CROPPED_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),

        #choose kernel size 32x32 pixels and stride length 32, which corresponds to a minimum of
        #200 unique strides / output filters
        #arbitrarily choose stride length equal to kernel dimensions
        Conv2D(200, (32,32), padding='valid'),
        Activation('relu')

])

optimizer = Adam(lr=1e-4)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[]
)

training_data, validation_data = partition_training_data([image_data, steering_angles], test_proportion=0.33)
training_images, training_steering = training_data
validation_images, validation_steering = validation_data

class SaveModel(KerasCallback):
    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if (epoch>9):
            with open ('model-' + str(epoch) + '.json', 'w') as file:
                file.write (model.to_json ())
                file.close ()

            model.save_weights ('model-' + str(epoch) + '.h5')


model.fit(
    x=training_images,
    steps_per_epoch=400*112,
    epochs=30,
    validation_data=(validation_images, validation_steering),
    callbacks=[SaveModel ()]
)

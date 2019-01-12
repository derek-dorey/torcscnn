from keras.models import Sequential

import properties as properties
import random as rand
import numpy as np
import cv2
import logging
import csv
from keras.layers import Dense, Activation, Reshape
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback as KerasCallback

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

IMAGE_DATA_DIRECTORY = 'C:/Users/Derek/Source/Repos/torcs-1.3.7/runtimed/images/'
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
    count = 0;

    for current_image in image_files:

        count += 1
        image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.IMREAD_GRAYSCALE)

        if image is None:
            del steering_angles[count]
            continue

        cropped_image = image[y_start:INPUT_IMAGE_HEIGHT, x_start:INPUT_IMAGE_WIDTH]
        downsize_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        image_data_set.append(downsize_image)

    return steering_angles, np.array(image_data_set)


'''
Partition training data into a training set and validation set
Args:
    training_data: list of lists of size two containing training data, e.g. [image_data_list, steering_angles_list]
    test_proportion: value between 0 and 1 that determines the index on which the data lists are split
Returns:
    1) train_set: [image_data_list, steering_angles_list] up to the index defined by the test_proportion
    2) valid_set: [image_data_list, steering_angles_list] of the remaining indices
'''
def partition_training_data(training_data, test_proportion):

    logging.info(' Partitioning training data... ')
    training_set_count = int((1.0-test_proportion)*len(training_data[0]))
    train_set = [list[0:training_set_count] for list in training_data]
    valid_set = [list[training_set_count:] for list in training_data]
    return train_set, valid_set


def augment_generator(images_arr, steering_arr, batch_size):

    logging.info(' Generating augmented data...')
    last_index = len(images_arr) - 1

    while 1:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):

            idx_img = rand.randint(0, last_index)
            im, steering = augment_record(images_arr[idx_img], steering_arr[idx_img])
            batch_img.append(im)
            batch_steering.append(steering)

        batch_img = np.asarray(batch_img)
        batch_steering = np.asarray(batch_steering)
        yield (batch_img, batch_steering)


def augment_record(im, steering):
    im = augment_image(im)
    if rand.uniform(0, 1) > 0.5:
        im = cv2.flip(im, 1)
        steering = - steering
    steering = steering + np.random.normal(0, 0.005)
    return im, steering


def augment_image(image):
    image = np.copy(image)

    (h, w) = image.shape[:2]

    # randomize brightness
    brightness = rand.uniform(-0.3, 0.3)
    image = np.add(image, brightness)

    # random squares
    rect_w = 25
    rect_h = 25
    rect_count = 30
    for i in range(rect_count):
        pt1 = (rand.randint(0, w), rand.randint(0, h))
        pt2 = (pt1[0] + rect_w, pt1[1] + rect_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)

    # rotation and scaling
    rot = 1
    scale = 0.02
    Mrot = cv2.getRotationMatrix2D((h / 2, w / 2), rand.uniform(-rot, rot), rand.uniform(1.0 - scale, 1.0 + scale))

    # affine transform and shifts
    pts1 = np.float32([[0, 0], [w, 0], [w, h]])
    a = 0
    shift = 2
    shiftx = rand.randint(-shift, shift);
    shifty = rand.randint(-shift, shift);
    pts2 = np.float32([[
        0 + rand.randint(-a, a) + shiftx,
        0 + rand.randint(-a, a) + shifty
    ], [
        w + rand.randint(-a, a) + shiftx,
        0 + rand.randint(-a, a) + shifty
    ], [
        w + rand.randint(-a, a) + shiftx,
        h + rand.randint(-a, a) + shifty
    ]])
    M = cv2.getAffineTransform(pts1, pts2)

    augmented = cv2.warpAffine(
        cv2.warpAffine(
            image
            , Mrot, (w, h)
        )
        , M, (w, h)
    )

    return augmented
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

training_data, validation_data = partition_training_data([image_data, steering_angles], test_proportion=0.33)
training_images, training_steering = training_data
validation_images, validation_steering = validation_data

batch_size = 112
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

model.fit_generator(
    augment_generator(training_images, training_steering, batch_size),
    steps_per_epoch=100*batch_size,
    epochs=epochs,
    validation_data=(validation_images, validation_steering),
    callbacks=[save_model]
)
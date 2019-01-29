from comet_ml import Experiment
from keras.models import Sequential
import keras.backend as backend

import properties as properties
import random as rand
import numpy as np
import cv2
import logging
import csv
import os
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Reshape
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback as KerasCallback
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

#experiment = Experiment(api_key="OPsq7RrD8Dl7fz8ne26NxQkcD",
#                        project_name="general", workspace="derek-dorey")

EDGE_DETECTION_GRAYSCALE = True

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

IMAGE_DATA_DIRECTORY = 'C:/Users/Paperspace/project/torcs-1.3.7/runtimed/'
INPUT_IMAGE_FORMAT = '.bmp'

STEERING_ANGLE_COLUMN = 'steer'
IMAGE_FILE_COLUMN = 'count'
IMAGE_FOLDER_COLUMN = 'imageFolder'

INPUT_IMAGE_WIDTH = 640
INPUT_IMAGE_HEIGHT = 480
CROPPED_IMAGE_HEIGHT = 320

STRAIGHT_DRIVING_INCLUSION_RATE = 1/(1-properties.STRAIGHT_DRIVING_EXCLUSION_FACTOR)

MODEL_OUTPUT_DIRECTORY = 'model_weights'

JSON_OUTPUT = 'model.json'
H5_OUTPUT = 'model.h5'


"""
Retrieve sensor data from torcs-1.3.7;
Discards most samples with minimal steering input according to STEERING_BOUNDARY and STRAIGHT_DRIVING_INCLUSION_RATE 
Returns:
    sensor_data_tuples: a list of tuples each containing pairs of:
        1) steering angle
        2) corresponding image path
"""
def load_sensor_data():

    logging.info(' Retrieving sensor data from ' +
                 'C:/Users/Paperspace/project/torcscnn/training_data/collated_sensor_data/' + properties.SENSOR_CSV_FILE)
    steering_data_list = []
    image_file_list = []

    with open('C:/Users/Paperspace/project/torcscnn/training_data/collated_sensor_data/' + properties.SENSOR_CSV_FILE) as sensor_data_csv:
        sensor_data_reader = csv.DictReader(sensor_data_csv, dialect="excel")

        #straight_driving_image_count = 0

        for row in sensor_data_reader:

            if row[STEERING_ANGLE_COLUMN] == '':
                continue

            try:
                steering_angle = float(row[STEERING_ANGLE_COLUMN])
                steering_data_list.append(steering_angle)
                image_file_list.append(row[IMAGE_FOLDER_COLUMN] + '/' + row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT)

            except ValueError:
                logging.WARNING('Invalid steering angle entry: ' + row[STEERING_ANGLE_COLUMN] + ' for image ' +
                                row[IMAGE_FILE_COLUMN] + '; Value excluded from training data.')

            #if abs(steering_angle) < properties.STEERING_BOUNDARY:

            #    straight_driving_image_count += 1

            #    if straight_driving_image_count == STRAIGHT_DRIVING_INCLUSION_RATE:

            #        steering_data_list.append(steering_angle)
            #        image_file_list.append(row[IMAGE_FILE_COLUMN] + INPUT_IMAGE_FORMAT)
            #       straight_driving_image_count = 0

            #else:

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
    error_count = 0

    for current_image in image_files:

        count += 1

        if current_image is None:
            del steering_angles[count]
            continue

        if EDGE_DETECTION_GRAYSCALE:

            og_image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.IMREAD_GRAYSCALE)
            normalized_image = np.subtract(np.divide(np.array(og_image).astype(np.float32), 255.0), 0.5)

            #cv2.imshow("Normalized Image", normalized_image)
            #cv2.waitKey(0)

            try:
                cropped_image = normalized_image[y_start:INPUT_IMAGE_HEIGHT, x_start:INPUT_IMAGE_WIDTH]
                #cv2.imshow("Cropped Image", cropped_image)
                #cv2.waitKey(0)
            except TypeError:
                del steering_angles[count]
                logging.warning(' ERRONEOUS ENTRY: ' + current_image)
                error_count += 1
                continue

            #edge_image = cv2.Canny(cropped_image, 50, 150)

            #cv2.imshow("Edge Detection", edge_image)
            #cv2.waitKey(0)

            downsize_image = cv2.resize(cropped_image, (0, 0), fx=0.5, fy=0.5)

            #cv2.imshow("Input", downsize_image)
            #cv2.waitKey(0)

            image_data_set.append(downsize_image)

        '''
        og_image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.IMREAD_GRAYSCALE)

        try:
            cropped_image = og_image[y_start:INPUT_IMAGE_HEIGHT, x_start:INPUT_IMAGE_WIDTH]
        except TypeError:
            del steering_angles[count]
            logging.warning(' ERRONEOUS ENTRY: ' + current_image)
            error_count += 1
            continue

        #cv2.imshow('cropped', cropped_image)
        #cv2.waitKey(0)

        downsize_image = cv2.resize(cropped_image, (0, 0), fx=0.5, fy=0.5)
        image_data_set.append(downsize_image)

        #cv2.imshow('og', og_image)
        #cv2.waitKey(0)
        #logging.info(' Loading image: ' + current_image)
        '''

        '''
        og_image = cv2.imread(IMAGE_DATA_DIRECTORY + current_image, cv2.COLOR_BGR2RGB)
        blurred_image = cv2.GaussianBlur(og_image, (3, 3), 0)
        edge_image = cv2.Canny(blurred_image, 50, 150)

        try:
            roi_image = roi(edge_image, [vertices])
        except TypeError:
            del steering_angles[count]
            logging.warning(' ERRONEOUS ENTRY: ' + current_image)
            error_count += 1
            continue


        cropped_image = roi_image[250:410, x_start:INPUT_IMAGE_WIDTH]
        downsize_image = cv2.resize(cropped_image, (0, 0), fx=0.5, fy=1)
        '''
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

        #image_data_set.append(downsize_image)

    return steering_angles, np.array(image_data_set)


steering_angles, image_paths = load_sensor_data()
steering_angles, image_data = load_images(steering_angles, image_paths)


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


optimizer = Adam(lr=0.001)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=[]
)

'''
def save_best_model(epoch, dir_path, num_ext, ext):
    tmp_file_name = os.listdir(dir_path)
    test = []
    num_element = -num_ext

    for x in range(0, len(tmp_file_name)):
        test.append(tmp_file_name[x][:num_element])
        float(test[x])

    lowest = min(test)

    return str(lowest) + ext
'''

epochs = 30

#predicted vs actual
'''
class SaveModel(KerasCallback):

    def on_epoch_end(self, epoch, logs={}):

        epoch += 1

        if epoch > 29:
            with open('model-' + str(epoch) + '.json', 'w') as file:
                file.write(model.to_json())
                file.close()

            model.save_weights('model-' + str(epoch) + '.h5')
'''

'''
class CollectOutputAndTarget(KerasCallback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        self.targets.append(backend.eval(self.var_y_true))
        self.outputs.append(backend.eval(self.var_y_pred))
        print(backend.eval(self.var_y_true))
        print(backend.eval(self.var_y_pred))


cbk = CollectOutputAndTarget()
fetches = [tf.assign(cbk.var_y_true, model.targets[0], validate_shape=False),
                tf.assign(cbk.var_y_pred, model.outputs[0], validate_shape=False)]
model._function_kwargs = {'fetches': fetches}
'''

checkpoint = ModelCheckpoint(filepath=MODEL_OUTPUT_DIRECTORY + '/{val_loss:.4f}.hdf5',
                             monitor='val_loss', verbose=0, save_best_only=True)
#save_model = SaveModel()

history_callback = model.fit(
    x=np.array(image_data),
    y=np.array(steering_angles),
    epochs=epochs,
    batch_size=64,
    validation_split=0.33,
    callbacks=[checkpoint]
)

'''
best_model = save_best_model(epochs, MODEL_OUTPUT_DIRECTORY, 5, '.hdf5')

logging.info(" Best model found: " + best_model)
logging.info(" Saving " + best_model + " as " + JSON_OUTPUT + ', ' + H5_OUTPUT)

model.load_weights(MODEL_OUTPUT_DIRECTORY + '/' + best_model)

with open(JSON_OUTPUT, 'w') as file:
    file.write(model.to_json())
    file.close()

model.save_weights(H5_OUTPUT)
'''
loss_history = history_callback.history["loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=',')

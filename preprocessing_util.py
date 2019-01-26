import glob
import properties as properties
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SINGLE_CSV = False
GENERATE_AUGMENT_CANDIDATES = False
GENERATE_COLLATED_DATA = True
LEFT_BOUNDARY = -0.5
RIGHT_BOUNDARY = abs(LEFT_BOUNDARY)
BIN_WIDTH = 0.01
SAMPLE_CEILING = 300
COLUMN_NAMES = ['Unnamed: 0', 'Unnamed: 0.1', '__doc__', '__init__', '__module__', 'accel', 'angle', 'brake', 'clutch',
                'count', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'distToMiddle', 'focus', 'fps', 'fuel',
                'gear', 'lastLapTime', 'posZ', 'racePos', 'rpm', 'speedX', 'speedY', 'speedZ', 'steer',
                'totalDistFromStart', 'wheelVel', 'imageFolder', 'Unnamed: 28']

steer_bins = np.arange(LEFT_BOUNDARY, RIGHT_BOUNDARY, BIN_WIDTH)

if SINGLE_CSV:
    sensor_data = pd.read_csv(properties.SENSOR_DATA_DIRECTORY + properties.SENSOR_CSV_FILE)
    #sensor_data = pd.read_csv('C:/Users/Paperspace/project/torcs-1.3.7/runtimed/Forza_Sensors.csv')
    #sensor_data = pd.read_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augmented_sensors.csv')
    sensor_data.head()

else:
    sensor_data_path = glob.glob(properties.SENSOR_DATA_DIRECTORY + '*.csv')
    sensor_data = pd.concat((pd.read_csv(f) for f in sensor_data_path))

sensor_data.steer.plot(title='Steering Data', fontsize=17, figsize=(10, 5), color= 'r')
plt.xlabel('Frames')
plt.ylabel('Steering Angle')
plt.show()

print("Dataset Size: ", len(sensor_data.steer))
plt.figure(figsize=(10, 8))
sensor_data.steer.hist(bins=steer_bins, color='r')
plt.xlabel('Steering Angle Bins')
plt.ylabel('Number of Samples')
plt.xlim()
plt.show()


sampled_data = pd.DataFrame(columns=COLUMN_NAMES)

if GENERATE_AUGMENT_CANDIDATES:
    augment_candidates = pd.DataFrame(columns=COLUMN_NAMES)

increment = 0


for sb in steer_bins:

    if increment % 2 == 0:

        try:
            downsized_sample = sensor_data[
                (sensor_data.steer >= sb) & (sensor_data.steer < (sb + BIN_WIDTH))].sample(n=SAMPLE_CEILING)

            sampled_data = pd.concat([sampled_data, downsized_sample])
            increment += 1

        except ValueError:

            downsized_sample = sensor_data[(sensor_data.steer >= sb) & (sensor_data.steer < (sb + BIN_WIDTH))]
            sampled_data = pd.concat([sampled_data, downsized_sample])

            if GENERATE_AUGMENT_CANDIDATES:
                augment_candidates = pd.concat([augment_candidates, downsized_sample])

            increment += 1
            continue

    else:

        try:
            downsized_sample = sensor_data[
                (sensor_data.steer > sb) & (sensor_data.steer <= (sb + BIN_WIDTH))].sample(n=SAMPLE_CEILING)

            sampled_data = pd.concat([sampled_data, downsized_sample])
            increment += 1

        except ValueError:

            downsized_sample = sensor_data[(sensor_data.steer > sb) & (sensor_data.steer <= (sb + BIN_WIDTH))]
            sampled_data = pd.concat([sampled_data, downsized_sample])

            if GENERATE_AUGMENT_CANDIDATES:
                augment_candidates = pd.concat([augment_candidates, downsized_sample])

            increment += 1
            continue

print("Sample Dataset Size: ", len(sampled_data.steer))
plt.figure(figsize=(10, 4))
sampled_data.steer.hist(bins=steer_bins, color='r')
plt.xlabel('Steering Angle Bins')
plt.ylabel('Number of Samples')
plt.show()

if GENERATE_COLLATED_DATA:
    sorted_data = sampled_data.sort_values(['imageFolder', 'count'], ascending=[True, True], kind='mergesort')
    sorted_data.to_csv('C:/Users/Paperspace/project/torcscnn/training_data/collated_sensor_data/collated_sensor_data.csv')
    print("Collated Dataset Size: ", len(sorted_data.steer))

if GENERATE_AUGMENT_CANDIDATES:
    sorted_data = augment_candidates.sort_values(['imageFolder', 'count'], ascending=[True, True], kind='mergesort')
    sorted_data.to_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augment_candidates.csv')
    print("Augmentation Candidates Size: ", len(sorted_data.steer))

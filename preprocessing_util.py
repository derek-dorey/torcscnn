import glob
import properties as properties
import pandas as pd
import matplotlib.pyplot as plt

SINGLE_CSV = False


if SINGLE_CSV:
    sensor_data = pd.read_csv(properties.SENSOR_DATA_DIRECTORY + properties.SENSOR_CSV_FILE)
    sensor_data.head()

else:
    sensor_data_path = glob.glob(properties.SENSOR_DATA_DIRECTORY + '*.csv')
    sensor_data = pd.concat((pd.read_csv(f) for f in sensor_data_path))

sensor_data.steer.plot(title='Steering Data Distribution', fontsize=17, figsize=(10,5), color= 'r')
plt.xlabel('Frames')
plt.ylabel('Steering Angle')
plt.show()

plt.figure(figsize=(10,8))
sensor_data.steer.hist(bins=100, color='r')
plt.xlabel('Steering Angle Bins')
plt.ylabel('Number of Samples')
plt.xlim()
plt.show()
print("Dataset Size: ", len(sensor_data.steer))

zero_steering = sensor_data[(sensor_data.steer > -properties.STEERING_BOUNDARY) &
                            (sensor_data.steer < properties.STEERING_BOUNDARY)].sample(frac=properties.STRAIGHT_DRIVING_EXCLUSION_FACTOR)
sensor_data = sensor_data.drop(zero_steering.index)

plt.figure(figsize=(10,4))
sensor_data.steer.hist(bins=100, color='r')
plt.xlabel('Steering Angle Bins')
plt.ylabel('Number of Samples')
plt.show()
print("Current Dataset Size: ", len(sensor_data.steer))
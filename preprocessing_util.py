import properties as properties
import pandas as pd
import matplotlib.pyplot as plt

sensor_data = pd.read_csv(properties.SENSOR_DATA_DIRECTORY + properties.SENSOR_CSV_FILE)
sensor_data.head()

sensor_data.steer.plot(title='Steering data distribution', fontsize=17, figsize=(10,5), color= 'r')
plt.xlabel('frames')
plt.ylabel('steering angle')
plt.show()

plt.figure(figsize=(10,8))
sensor_data.steer.hist(bins=100, color='r')
plt.xlabel('steering angle bins')
plt.ylabel('counts')
plt.xlim()
plt.show()
print("Dataset Size: ", len(sensor_data.steer))

zero_steering = sensor_data[(sensor_data.steer > -properties.STEERING_BOUNDARY) &
                            (sensor_data.steer < properties.STEERING_BOUNDARY)].sample(frac=properties.STRAIGHT_DRIVING_EXCLUSION_FACTOR)
sensor_data = sensor_data.drop(zero_steering.index)

plt.figure(figsize=(10,4))
sensor_data.steer.hist(bins=100, color='r')
plt.xlabel('steering angle bins')
plt.ylabel('counts')
plt.show()
print("Current Dataset Size: ", len(sensor_data.steer))
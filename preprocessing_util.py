import pandas as pd
import matplotlib.pyplot as plt

SENSOR_DATA_FILE = 'C:/Users/Derek/Source/Repos/torcs-1.3.7/runtimed/sensors.csv'

sensor_data = pd.read_csv(SENSOR_DATA_FILE)
sensor_data.head()

sensor_data.steer.plot(title='Steering data distribution', fontsize=17, figsize=(10,5), color= 'r')
plt.xlabel('frames')
plt.ylabel('steering angle')
plt.show()

plt.figure(figsize=(10,8))
sensor_data.steer.hist(bins=1000, color='r')
plt.xlabel('steering angle bins')
plt.ylabel('counts')
plt.xlim()
plt.show()
print("Dataset Size: ", len(sensor_data.steer))

zero_steering = sensor_data[(sensor_data.steer > - 0.00005) & (sensor_data.steer < 0.00005)].sample(frac=0.8)
sensor_data = sensor_data.drop(zero_steering.index)

plt.figure(figsize=(10,4))
sensor_data.steer.hist(bins=100, color='r')
plt.xlabel('steering angle bins')
plt.ylabel('counts')
plt.show()
print("Current Dataset Size: ", len(sensor_data.steer))
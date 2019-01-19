SENSOR_DATA_DIRECTORY = 'C:/Users/Paperspace/project/torcscnn/sensor_data/'
SENSOR_CSV_FILE = 'E_Track_6_Sensors.csv'

# [-STEERING_BOUNDARY < x < STEERING_BOUNDARY] determines whether image with steering angle 'x'
# is categorized as driving straight
STEERING_BOUNDARY = 0.01

# STRAIGHT_DRIVING_EXCLUSION_FACTOR determines the proportion of straight driving images to be excluded
# (e.g. a value of 0.8 will exclude 80% of the images categorized as straight driving)
STRAIGHT_DRIVING_EXCLUSION_FACTOR = 0.8

import pandas as pd

OG_CSV = 'Wheel_1_Sensors.csv'
IMAGE_FOLDER = 'Wheel_1_Images'
NEW_CSV = 'Wheel_1_Sensors_modified.csv'

sensor_data = pd.read_csv('C:/Users/Paperspace/project/torcscnn/sensor_data/' + OG_CSV)
sensor_data.head()

sensor_data.insert(loc=28, column='imageFolder', value=IMAGE_FOLDER, allow_duplicates=True)
sensor_data.to_csv('C:/Users/Paperspace/project/torcscnn/sensor_data/' + NEW_CSV)

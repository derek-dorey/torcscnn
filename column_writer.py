import pandas as pd

OG_CSV = 'Ruudskogen_Sensors.csv'
IMAGE_FOLDER = 'Ruudskogen_Images'
NEW_CSV = 'Ruudskogen_Sensors_modified.csv'

sensor_data = pd.read_csv('C:/Users/Paperspace/project/torcscnn/sensor_data/' + OG_CSV)
sensor_data.head()

sensor_data.insert(loc=28, column='imageFolder', value=IMAGE_FOLDER, allow_duplicates=True)
sensor_data.insert(loc=29, column='Unnamed: 28', value='', allow_duplicates=True)
sensor_data.to_csv('C:/Users/Paperspace/project/torcscnn/training_data/' + NEW_CSV)

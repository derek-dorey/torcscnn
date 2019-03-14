import pandas as pd

OG_CSV = 'Ruudskogen_Recovery_Sensors.csv'
IMAGE_FOLDER = 'Ruudskogen_Recovery_Images'
NEW_CSV = 'Ruudskogen_Recovery_Sensors_modified.csv'

sensor_data = pd.read_csv('C:/Users/Paperspace/project/torcs-1.3.7/runtimed/' + OG_CSV)
sensor_data.head()

sensor_data.insert(loc=28, column='imageFolder', value=IMAGE_FOLDER, allow_duplicates=True)
sensor_data.insert(loc=29, column='Unnamed: 28', value='', allow_duplicates=True)
sensor_data.to_csv('C:/Users/Paperspace/project/torcscnn/training_data/' + NEW_CSV)

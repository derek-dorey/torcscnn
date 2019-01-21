import pandas as pd
from PIL import Image, ImageOps

IMAGE_DATA_DIRECTORY = 'C:/Users/Paperspace/project/torcs-1.3.7/runtimed/'
AUGMENTED_IMAGE_DIRECTORY = 'augmented_images'

augment_candidates = pd.read_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augment_candidates.csv')
augment_candidates.head()

for index, row in augment_candidates.iterrows():

    og_image_folder = row['imageFolder'] + '/'
    og_image_file = str(row['count']) + '.bmp'

    try:
        og_image = Image.open(IMAGE_DATA_DIRECTORY + og_image_folder + og_image_file)
    except FileNotFoundError:
        print(row['steer'])
        continue

    #og_image.show()

    flipped_image = ImageOps.mirror(og_image)

    #flipped_image.show()

    #if row['steer'] < -0.3:
    #    print(row['steer'])

    augment_candidates.at[index, 'imageFolder'] = AUGMENTED_IMAGE_DIRECTORY

    new_image_file = '0' + str(row['count'])
    augment_candidates.at[index, 'count'] = new_image_file
    new_steer = -1 * row['steer']

    augment_candidates.at[index, 'steer'] = new_steer

    #if row['steer'] > 0.3:
    #    print(row['steer'])

    flipped_image.save(IMAGE_DATA_DIRECTORY + '/' + AUGMENTED_IMAGE_DIRECTORY + '/' + new_image_file + ".bmp")


augment_candidates.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1, inplace=True)
augment_candidates.to_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augmented_sensors.csv')


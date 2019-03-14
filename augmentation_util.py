import pandas as pd
import cv2
import numpy as np
import random as rand
import scipy.misc
from PIL import Image, ImageOps

IMAGE_DATA_DIRECTORY = 'C:/Users/Paperspace/project/torcs-1.3.7/runtimed/'
AUGMENTED_IMAGE_DIRECTORY = 'Augmented_Images'

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

    flipped_image = ImageOps.mirror(og_image)

    image_source_folder = row['imageFolder']

    if image_source_folder == 'CG_Track_2_Images':
        prefix = '100000'

    elif image_source_folder == 'CG_Track_2_Recovery_Images':
        prefix = '200000'

    elif image_source_folder == 'CG_Track_3_Images':
        prefix = '300000'

    elif image_source_folder == 'CG_Track_3_Recovery_Images':
        prefix = '400000'

    elif image_source_folder == 'E_Track_4_Images':
        prefix = '500000'

    elif image_source_folder == 'E_Track_4_Recovery_Images':
        prefix = '600000'

    elif image_source_folder == 'E_Track_6_Images':
        prefix = '700000'

    elif image_source_folder == 'E_Track_6_Recovery_Images':
        prefix = '800000'

    elif image_source_folder == 'Forza_Images':
        prefix = '900000'

    elif image_source_folder == 'Forza_Recovery_Images':
        prefix = '1000000'

    elif image_source_folder == 'Wheel_1_Images':
        prefix = '11000000'

    elif image_source_folder == 'Wheel_1_Recovery_Images':
        prefix = '12000000'

    elif image_source_folder == 'Ruudskogen_Images':
        prefix = '13000000'

    elif image_source_folder == 'Ruudskogen_Recovery_Images':
        prefix = '14000000'

    else:
        prefix = '15000000'

    augment_candidates.at[index, 'imageFolder'] = AUGMENTED_IMAGE_DIRECTORY

    new_image_file = prefix + str(row['count'])
    augment_candidates.at[index, 'count'] = new_image_file
    new_steer = -1 * row['steer']

    augment_candidates.at[index, 'steer'] = new_steer

    #if row['steer'] > 0.3:
    #    print(row['steer'])

    flipped_image.save(IMAGE_DATA_DIRECTORY + '/' + AUGMENTED_IMAGE_DIRECTORY + '/' + new_image_file + ".bmp")


augment_candidates.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1, inplace=True)
augment_candidates.to_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augmented_sensors.csv')


def augment_image(image):
    image = np.copy(image)

    (h, w) = image.shape[:2]

    # rotation and scaling
    rot = 1
    scale = 0.02
    Mrot = cv2.getRotationMatrix2D((h / 2, w / 2), rand.uniform(-rot, rot), rand.uniform(1.0 - scale, 1.0 + scale))

    # affine transform and shifts
    pts1 = np.float32([[0, 0], [w, 0], [w, h]])
    a = 0
    shift = 2
    shiftx = rand.randint(-shift, shift);
    shifty = rand.randint(-shift, shift);
    pts2 = np.float32([[
        0 + rand.randint(-a, a) + shiftx,
        0 + rand.randint(-a, a) + shifty
    ], [
        w + rand.randint(-a, a) + shiftx,
        0 + rand.randint(-a, a) + shifty
    ], [
        w + rand.randint(-a, a) + shiftx,
        h + rand.randint(-a, a) + shifty
    ]])
    M = cv2.getAffineTransform(pts1, pts2)

    augmented = cv2.warpAffine(
        cv2.warpAffine(
            image
            , Mrot, (w, h)
        )
        , M, (w, h)
    )

    return augmented


augment_candidates_2 = pd.read_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augment_candidates.csv')
augment_candidates_2.head()
augmented_images_2_count = 0

for index, row in augment_candidates_2.iterrows():

    og_image_folder = row['imageFolder'] + '/'
    og_image_file = str(row['count']) + '.bmp'

    try:
        cv2_image = cv2.imread(IMAGE_DATA_DIRECTORY + og_image_folder + og_image_file, cv2.IMREAD_UNCHANGED)
    except FileNotFoundError:
        print(row['steer'])
        continue

    augmented_image = augment_image(cv2_image)

    image_source_folder = row['imageFolder']

    augment_candidates_2.at[index, 'imageFolder'] = 'Augmented_Images_2'

    new_image_file = str(augmented_images_2_count)
    augmented_images_2_count += 1
    augment_candidates_2.at[index, 'count'] = new_image_file

    new_steer = (row['steer'] + np.random.normal(0, 0.0005))

    augment_candidates_2.at[index, 'steer'] = new_steer

    scipy.misc.imsave(IMAGE_DATA_DIRECTORY + '/Augmented_Images_2/' + new_image_file + ".bmp", augmented_image)


augment_candidates_2.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1, inplace=True)
augment_candidates_2.to_csv('C:/Users/Paperspace/project/torcscnn/augmented_data/augmented_sensors_2.csv')

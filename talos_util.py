import pandas as pd
import matplotlib.pyplot as plt

TALOS_DIRECTORY = 'C:/Users/Paperspace/project/Talos/'
TALOS_CSV = 'N500_NORMALIZED_GRAYSCALE.csv'
SORTED_CSV = 'SORTED_N500_NORMALIZED_GRAYSCALE.csv'

COLUMN_NAMES = ['Unnamed: 0', 'round_epochs', 'val_loss', 'loss', 'lr', 'first_layer', 'validation_split', 'batch_size',
                'epochs', 'dropout', 'optimizer', 'loss', 'last_activation', 'weight_regulizer']

GENERATE_SORTED_CSV = False

LEARNING_RATE = False
FIRST_LAYER = False
VALIDATION_SPLIT = False
BATCH_SIZE = False
DROPOUT = True

talos_data = pd.read_csv(TALOS_DIRECTORY + TALOS_CSV)

sorted_data = talos_data.sort_values(['lr', 'first_layer', 'validation_split', 'batch_size', 'dropout'],
                                     ascending=[True, True, True, True, True], kind='mergesort')

output = pd.DataFrame(columns=COLUMN_NAMES)

if GENERATE_SORTED_CSV:
    sorted_data.to_csv(TALOS_DIRECTORY + SORTED_CSV)


FIRST_ITERATION = True
index = 0

for index, row in sorted_data.iterrows():

    learning_rate = row['lr']
    first_layer = row['first_layer']
    validation_split = row['validation_split']
    batch_size = row['batch_size']
    dropout = row['dropout']

    if FIRST_ITERATION:

        first_lr = learning_rate
        first_first_layer = first_layer
        first_validation_split = validation_split
        first_batch_size = batch_size
        first_dropout = dropout
        FIRST_ITERATION = False
        output.loc[index] = row
        index += 1
        continue

    if LEARNING_RATE and (first_layer == first_first_layer) and (validation_split == first_validation_split) and \
            (batch_size == first_batch_size) and (dropout == first_dropout):

        output.loc[index] = row
        index += 1

    elif FIRST_LAYER and (learning_rate == first_lr) and (validation_split == first_validation_split) and \
            (batch_size == first_batch_size) and (dropout == first_dropout):

        output.loc[index] = row
        index += 1

    elif VALIDATION_SPLIT and (learning_rate == first_lr) and (first_layer == first_first_layer) and \
            (batch_size == first_batch_size) and (dropout == first_dropout):

        output.loc[index] = row
        index += 1

    elif BATCH_SIZE and (learning_rate == first_lr) and (first_layer == first_first_layer) and \
            (validation_split == first_validation_split) and (dropout == first_dropout):

        output.loc[index] = row
        index += 1

    elif DROPOUT and (learning_rate == first_lr) and (first_layer == first_first_layer) and \
            (validation_split == first_validation_split) and (batch_size == first_batch_size):

        output.loc[index] = row
        index += 1


if LEARNING_RATE:

    output.to_csv(TALOS_DIRECTORY + 'learning_rate.csv')

if FIRST_LAYER:

    output.to_csv(TALOS_DIRECTORY + 'first_layer.csv')

if VALIDATION_SPLIT:

    output.to_csv(TALOS_DIRECTORY + 'validation_split.csv')

if BATCH_SIZE:

    output.to_csv(TALOS_DIRECTORY + 'batch_size.csv')

if DROPOUT:

    output.to_csv(TALOS_DIRECTORY + 'dropout.csv')


talos_data.val_loss.plot(title='Loss', fontsize=17, figsize=(10, 5), color= 'b')
plt.xlabel('Model')
plt.ylabel('Val_Loss')
plt.show()
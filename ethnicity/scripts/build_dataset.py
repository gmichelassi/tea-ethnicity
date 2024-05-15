import pandas as pd
import shutil
import os

BASE_PATH = 'ethnicity/data/'
SOURCE_DATASET_FOLDER = 'fairface-img-margin025-trainval'
TARGET_DATASET_FOLDER = 'fairface-ethnicity'


def build():
    train_dataset_info = pd.read_csv('ethnicity/data/fairface_label_train.csv')
    validation_dataset_info = pd.read_csv('ethnicity/data/fairface_label_val.csv')

    move_images(train_dataset_info)
    move_images(validation_dataset_info)


def move_images(dataset: pd.DataFrame):
    for _, row in dataset.iterrows():
        image_file_name = row['file']
        image_class = row['race']
        target_folder = image_file_name.split('/')[-2]
        target_file_name = image_file_name.split('/')[-1]

        source_path = f'{BASE_PATH}{SOURCE_DATASET_FOLDER}/{image_file_name}'
        base_target_path = f'{BASE_PATH}{TARGET_DATASET_FOLDER}/{target_folder}/{image_class}'
        target_path = f'{base_target_path}/{target_file_name}'

        print(f'Moving image from {source_path} to {target_path}')

        if not os.path.exists(base_target_path):
            os.makedirs(base_target_path)

        try:
            shutil.move(source_path, target_path)
        except FileNotFoundError as e:
            print(f'Error: {e}')

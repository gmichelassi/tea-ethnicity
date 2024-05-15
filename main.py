import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Sequential

from ethnicity.network.layers import PCALayer


DATASET_FOLDER = 'ethnicity/data/fairface-ethnicity/'


def process_images():
    batch_size = 32
    img_height = 224
    img_width = 224

    training_dataset, validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_FOLDER,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        validation_split=0.2,
        subset='both',
        shuffle=True,
        seed=97623,
    )

    class_names = training_dataset.class_names
    num_classes = len(class_names)

    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(2, 9, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(2, 9, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(2, 9, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        PCALayer(number_of_components=10),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=3,
        batch_size=batch_size
    )


def dataset_info():
    for folder in os.listdir(DATASET_FOLDER):
        folder_path = os.path.join(DATASET_FOLDER, folder)

        if os.path.isdir(folder_path):
            print(f'Folder: {folder} - Number of files: {len(os.listdir(folder_path))}')


def dataset_visualization(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(dataset.class_names[np.argmax(labels[i])])
            plt.axis("off")

    plt.show()


if __name__ == '__main__':
    process_images()

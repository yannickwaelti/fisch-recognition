from math import ceil

from tensorflow import keras
from tensorflow.keras import layers


def get_loader(train_dir, batch_size, image_size):
    print("Getting loaders")
    trn_transforms = keras.Sequential([
        layers.experimental.preprocessing.Resizing(ceil(image_size*1.2), ceil(image_size*1.2)),
        layers.experimental.preprocessing.RandomCrop(image_size, image_size)
    ])

    val_transforms = keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_size, image_size)
    ])

    trn_dataset = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="training",
        seed=125,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    val_dataset = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="validation",
        seed=125,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    labels = trn_dataset.class_names
    trn_dataset = trn_dataset.map(lambda x, y: (trn_transforms(x), y))
    val_dataset = val_dataset.map(lambda x, y: (val_transforms(x), y))

    return trn_dataset, val_dataset, labels

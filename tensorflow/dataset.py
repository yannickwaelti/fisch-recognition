import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_loader(train_dir, dev_dir, batch_size, image_size):
    print("Getting loaders")
    trn_transforms = keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.experimental.preprocessing.Resizing(300, 300),
        layers.experimental.preprocessing.RandomCrop(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomFlip("vertical")
    ])

    val_transforms = keras.Sequential([
        layers.experimental.preprocessing.Resizing(image_size, image_size)
    ])

    trn_dataset = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    val_dataset = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    trn_dataset = trn_dataset.map(lambda x, y: (trn_transforms(x), y))
    val_dataset = val_dataset.map(lambda x, y: (val_transforms(x), y))

    return trn_dataset, val_dataset

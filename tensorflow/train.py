import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from dataset import get_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

URL = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 484
NUM_EPOCHS = 20
INITIAL_LEARNING_RATE = 0.1
FINAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = INITIAL_LEARNING_RATE / FINAL_LEARNING_RATE / NUM_EPOCHS
DATA_DIR = "dataset/images/by_class"
MODEL_PATH = "efficientb0/"
LOAD_MODEL = False
USE_KERAS_FUNCTIONS = True

img_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.experimental.preprocessing.RandomFlip(),
        layers.experimental.preprocessing.RandomContrast(0.05)
    ],
    name="img_augmentation"
)

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
os.environ["TFHUB_CACHE_DIR"] = "tfhub_cache/"


def get_model(url, img_size, num_classes):
    model = keras.Sequential([
        img_augmentation,
        hub.KerasLayer(url, trainable=True),
        layers.Dense(1000, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.build([None, img_size, img_size, 3])
    return model


def get_keras_model(img_size, num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    model = keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    return model


@tf.function
def train_step(data, labels, acc_metric, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric.update_state(labels, predictions)


def evaluate_model(ds_validation, model):
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
    for idx, (data, labels) in enumerate(ds_validation):
        y_pred = model(data, training=False)
        accuracy_metric.update_state(labels, y_pred)

    accuracy = accuracy_metric.result()
    print(f"Accuracy over validation set: {accuracy}")


def lr_time_based_decay(epoch, learning_rate):
    return learning_rate / (1 + LEARNING_RATE_DECAY * epoch)


def visualize_data(train_dataset, labels):
    fig = plt.figure()
    for tmp, (image, label) in enumerate(train_dataset.take(1)):
        for i in range(9):
            fig.add_subplot(3, 3, i + 1)
            plt.imshow(image[i].numpy().astype("uint8"))
            plt.title("{}".format(labels[label[i]]))
            plt.axis("off")
    fig.show()


def visualize_augmented_data(train_dataset, labels):
    fig = plt.figure()
    for tmp, (image, label) in enumerate(train_dataset.take(1)):
        for i in range(9):
            fig.add_subplot(3, 3, i + 1)
            aug_img = img_augmentation(tf.expand_dims(image[i], axis=0))
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.title("{}".format(labels[label[i]]))
            plt.axis("off")
    fig.show()


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def main():
    train_loader, dev_loader, labels = get_loader(DATA_DIR, BATCH_SIZE, IMG_SIZE)
    visualize_data(train_loader, labels)

    visualize_augmented_data(train_loader, labels)

    optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay([10, 15],
                                                                        [INITIAL_LEARNING_RATE,
                                                                         0.01,
                                                                         LEARNING_RATE_DECAY]))
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    if LOAD_MODEL:
        print("Loading Model")
        model = keras.models.load_model(MODEL_PATH)

    elif USE_KERAS_FUNCTIONS:
        print("Using Keras functions")
        print("Building Model")
        model = get_keras_model(IMG_SIZE, NUM_CLASSES)
        keras_training_function(train_loader, dev_loader, model, optimizer, loss_fn)

    else:
        print("Building Model")
        model = get_model(URL, IMG_SIZE, NUM_CLASSES)
        custom_training_function(train_loader, dev_loader, model, optimizer, loss_fn)

    model.save(MODEL_PATH)
    print(model.predict(dev_loader))


def keras_training_function(train_loader, dev_loader, model, optimizer, loss):
    model.compile(
        optimizer=optimizer, loss=loss, metrics=["accuracy"]
    )

    model.summary()

    epochs = NUM_EPOCHS  # @param {type: "slider", min:10, max:100}
    hist = model.fit(train_loader, epochs=epochs, validation_data=dev_loader, verbose=2)
    plot_hist(hist)


def custom_training_function(train_loader, dev_loader, model, optimizer, loss):
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for idx, (data, labels) in enumerate(tqdm(train_loader)):
            train_step(data, labels, acc_metric, model, loss, optimizer)

            if idx % 50 == 0 and idx > 0:
                train_acc = acc_metric.result()
                print(f"Accuracy over partial epoch: {train_acc}")

                evaluate_model(dev_loader, model)
                model.save(MODEL_PATH)


if __name__ == "__main__":
    main()

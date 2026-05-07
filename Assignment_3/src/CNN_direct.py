#Import libraries
import os
import argparse

# Data tools
import numpy as np
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tensorflow tools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input 
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPool2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.optimizers import SGD

# Define an argument parser that allows user to 
# pass arguments, choosing number of training epochs and the data to 
# run the script on 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d",
                        required=True,
                        help="Choose either the base or cropped folder")
    parser.add_argument("--epochs", "-e",
                        required=False,
                        default=10,
                        type=int,
                        help="Number of training iterations")
    return parser.parse_args()

def load_image_dataset(
    directory,
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    seed=42):

    # Load training images from directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training"
    )

    # Load validation images using the same split settings
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation"
    )

    # Extract class names from the training dataset
    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


# Extract accuracy and loss values from training history
def plot_learning_curves(history, save_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Rescaling(1./255))

    # CONV => ReLU
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))

    # FC classifier
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"]
    )

    return model

# Train the model and record accuracy/loss at each epoch
def train_model(model, train_ds, val_ds, epochs=10):
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=epochs,
        verbose=1
    )
    return history

def evaluate_model(model, val_ds, class_names, output_folder, data_name):

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    report_path = os.path.join(output_folder, f"{data_name}_classification_report.txt")
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(report_path, "w") as f:
        f.write(report)

    print(report)

def main():

    args = parse_args()

    DATA_DIR = f"../data/lego/{args.data}"

    train_ds, val_ds, class_names = load_image_dataset(DATA_DIR)

    model = build_model(
        input_shape=(256, 256, 3),
        num_classes=len(class_names)
    )

    model.summary()

    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../output")
    os.makedirs(output_folder, exist_ok=True)
    
    plot_learning_curves(history, save_path=os.path.join(output_folder, f"{args.data}_direct_learning_curves.png"))

    evaluate_model(
        model,
        val_ds,
        class_names,
        output_folder,
        f"{args.data}_direct"
    )

if __name__ == "__main__":
    main()
# Tensorflow tools
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import (preprocess_input, VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)
from tensorflow.keras.models import Model

#scikit-learn
from sklearn.metrics import classification_report

#General tools
import numpy as np
import matplotlib.pyplot as plt
import os 
import argparse

# VGG expects images of size 224x224
IMAGE_SIZE = (224, 224)

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
    image_size=IMAGE_SIZE,
    batch_size=32,
    validation_split=0.2,
    seed=42):

    #Define settings for both training and validation data
    shared = dict(
        labels='inferred',
        label_mode='categorical', 
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split
    )

    # Load training split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        subset="training", **shared)

    # Load validation split
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        subset="validation", **shared)
    
    # Extract class names from subfolder name
    class_names = train_ds.class_names

    # Convert images to VGG16's desired format
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    return train_ds, val_ds, class_names

# Load VGG16 excluding the original classifier
# Freeze layers so that weights are not updated during training
def build_model(num_classes, image_size=IMAGE_SIZE):
    base_model = VGG16(include_top=False,
                       pooling='avg',
                       input_shape=(*image_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # add new classifier head
    flat = Flatten()(base_model.output)
    dense = Dense(128, activation='relu')(flat)
    batchnorm = BatchNormalization()(dense)
    output = Dense(num_classes, activation='softmax')(batchnorm)

    #Combine VGG16 base with the new classifier head
    model = Model(inputs=base_model.inputs, outputs=output)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train model and save accuracy and loss for each epoc 
def train_model(model, train_ds, val_ds, epochs=10):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    return history

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

def evaluate_model(model, val_ds, class_names, output_folder, data_name):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1)) 
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    report_path = os.path.join(output_folder, f"{data_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

def main():

    args = parse_args()

    DATA_DIR = f"../data/lego/{args.data}"

    train_ds, val_ds, class_names = load_image_dataset(DATA_DIR)

    model = build_model(num_classes=len(class_names))
    model.summary()

    history = train_model(model, train_ds, val_ds, epochs=args.epochs)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../output")
    os.makedirs(output_folder, exist_ok=True)

    plot_learning_curves(history,
                         save_path=os.path.join(output_folder, f"{args.data}_vgg16_learning_curves.png"))
    evaluate_model(model, val_ds, class_names, output_folder, f"{args.data}_vgg16")

if __name__ == "__main__":
    main()
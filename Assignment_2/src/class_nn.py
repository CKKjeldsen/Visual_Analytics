import os
import cv2
import numpy as np
import argparse

# Import dataset 
from tensorflow.keras.datasets import cifar10
# Machine learning tools
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
# Visualization tools
from matplotlib import pyplot as plt

# Define an argument parser that allows user to 
# pass arguments, choosing number of training epochs and the 
# number of nodes in the hidden layer
def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e",
                    required=False,
                    default=10,
                    type=int,
                    help="Number of training iterations")

    parser.add_argument("--hidden_layers", "-hl",
                    required=False,
                    default=10,
                    type=int,
                    help="Number of nodes in the hidden layer")

    args = parser.parse_args()

    return args

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #define labels
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]

    #Convert to grayscale 
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    #Scaling to between 0 and 1
    X_train_scaled = X_train_grey/255.0
    X_test_scaled = X_test_grey/255.0

    #Reshape from 3d to 2d, going from being 32x32 to 1x1024
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples, nx * ny))

    nsamples_test, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples_test, nx * ny))

    return (X_train_dataset, y_train), (X_test_dataset, y_test), labels


def main():
    args = user_input()

    (X_train, y_train), (X_test, y_test), labels = load_data()

    #Gives some information, and tests that that the script works by printing in the console
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    clf = MLPClassifier(random_state=42,
                    hidden_layer_sizes=(args.hidden_layers, ),
                    solver="adam",
                    learning_rate="adaptive",
                    early_stopping=True,
                    verbose=True,
                    max_iter=args.epochs).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)

    os.makedirs("../output", exist_ok=True)
    
    with open(f"../output/classification_report_nn_e{args.epochs}_hl{args.hidden_layers}.txt", "w") as f:
        f.write(report)

    plt.plot(clf.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.savefig(f"../output/loss_curve_nn_e{args.epochs}_hl{args.hidden_layers}.png")
    plt.close()
    
if __name__ == "__main__":
    main()
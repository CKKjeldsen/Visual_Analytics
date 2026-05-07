import os
import cv2
import numpy as np

# Import dataset  
from tensorflow.keras.datasets import cifar10
# Machine learning tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
    
    (X_train, y_train), (X_test, y_test), labels = load_data()

    #Gives some information, and tests that that the script works by printing in the console
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    clf = LogisticRegression(tol=0.1,
                        verbose=False,
                        solver="saga").fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)

    os.makedirs("../output", exist_ok=True)
    
    with open("../output/classification_report_logistic.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
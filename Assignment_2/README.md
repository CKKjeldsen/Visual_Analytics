# Assignment 2: Classification benchmarks with logistic regression and neural networks 

### Christian K. Kjeldsen

This project repository contains two python scripts that create baselines for classifying images in the CIFAR-10 dataset. The first script, class_log.py, does this for a logistic regression classifier, while the second script, class_nn.py, does this for a neural network classifier. Therefore, the performance of the two can be compared to evaluate performance.  

## Repo structure 

```
Assignment_2/
├── output/
│   ├── classification_report_logistic.txt
│   ├── classification_report_nn_e100_hl256.txt
│   └── loss_curve_nn_e100_hl256.png
├── src/
│   ├── class_log.py
│   └── class_nn.py
└── requirements.txt
└── setup.sh
└── README.md
```

## Data

The data used in this project is available from: 

https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is presented in detail in the technical report: *Learning Multiple Layers of Features from Tiny Images*, Alex Krizhevsky, 2009.
Since the dataset is contained in the tensorflow package, it is loaded with:

```python
from tensorflow.keras.datasets import cifar10 
```

And the repository does therefore not contain a data/ folder. 

## Requirements

This project is coded in Python version 3.12.3. The included setup.sh script creates a virtual environment and installs all dependencies:

```bash
bash setup.sh
```
This runs the following steps:

```
python3 -m venv env
source ./env/bin/activate
pip install -r requirements.txt
deactivate
```

And the exact requirements for this project are in the requirements.txt, showing:

matplotlib==3.10.8

numpy==2.4.4

scikit_learn==1.8.0

tensorflow==2.21.0

cv2==4.13.0

## Usage

Both scripts can be run from the command line in a bash terminal. First change directory to the src/ folder. The class_log.py script does not take any arguments, while the class_nn.py takes two arguments form the command line, number of training epochs (default 10) and number of nodes in the single hidden layer (default 10).

### class_log.py example

```bash
python class_log.py #runs the script
```

### class_nn.py examples

```bash
python class_nn.py #runs the script 
#with 10 epochs and 10 nodes in the hidden layer

python class_nn.py --epochs 100 -hl 256 #runs the 
#script with 100 epochs and 256 nodes in the hidden layer
```

## Output

Running the class_log.py saves a single file to the output/ folder, a classification report (classification_report_log.txt). Running the class_nn.py saves two files to the output/ folder, a classification report and a training loss curve. The files produced by the class_nn.py script appends the passed arguments in the file names, for example classification_report_nn_e100_hl256.txt for a run with 100 epochs and 256 nodes in the hidden layer. 

Running both scripts once will produce a total of 3 output files. 

### Key points from the output files 

The class_log.py script had a 31% accuracy and a f1-score of 0.31, showing relatively poor overall performance. This model performed best on images of machines, having a f1-score of 0.41 on trucks and 0.38 on ships, while it performed most poorly on images of animals, with a f1-score of 0.17 on cats and 0.23 on deer. 

The class_nn.py script run with 100 epochs and 256 nodes in the hidden layer performed better with an overall accuracy of 42% and a f1-score of 0.42. 
This model also saw best performance on machines (ship f1-score: 0.53; truck f1-score: 0.50) and the poorest performance on animals (cat f1-score: 0.25; deer f1-score: 0.34). 

## Discussion and limitations 

The learning curve plot for the class_nn.py script shows that the model initially learns quickly, and the curve continues to decrease throughout training. The model has stopped early after the 50th epoch because it was no longer improving. 

These results show that a neural network model with a single hidden layer outperforms a logistic classifier by about 11 percentage points. Since there are 10 classes, randomly assigning labels would give an accuracy of 10%, so the logistic classfier performs about 3 times as well as random chance while the neural network model works about 4 times as well.  

The performance of the class_nn.py script could probably be improved substantially be using more hidden layers. It is still limited by the fact that color is removed from the images, meaning that the model has access to less information than is available in the original images. Likewise, the spatial relations between pixels is destroyed by flattening the images. These circumstances speak in favour of using a convolutional neural network for image classification tasks instead. 
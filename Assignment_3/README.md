# Assignment 3: LEGO brick classification with CNNs 

### Christian K. Kjeldsen

This project repository contains two python scripts that classify images of LEGO bricks. The first script, CNN_direct.py, is built from the bottom up, while the second script, CNN_VGG16.py, utilizes transfer learning by using VGG16 as a pretrained feature extractor. Therefore, the performance of the two can be compared to illustrate the benefits of using transfer learning. 

## Repo structure

```
Assignment_3/
├── data/
│   └── lego/
│       ├── base/
│       │   └── [Brick_1x1]/
│       │       └── [~100+ images per class]
│       │   └── [...]/
│       │       └── [...]
│       └── cropped/
│       │   └── [Brick_1x1]/
│       │       └── [~100+ images per class]
│       │   └── [...]/
│       │       └── [...]
├── output/
│   ├── base_direct_classification_report.txt
│   ├── base_direct_learning_curves.png
│   ├── base_vgg16_classification_report.txt
│   ├── base_vgg16_learning_curves.png
│   ├── cropped_direct_classification_report.txt
│   ├── cropped_direct_learning_curves.png
│   ├── cropped_vgg16_classification_report.txt
│   └── cropped_vgg16_learning_curves.png
└── src/
    └── CNN_direct.py
    └── CNN_VGG16.py
└── requirements.txt
└── setup.sh
└── README.md
```

## Data

The data used in this project is available from: 

https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition

which is licensed under CC BY-SA 3.0. The dataset contains 9160 image files in three folder layers. Each image is in a folder named after the type of brick, e.g. Plate_1x1_Slope/, and each of these folders are placed in one of two dataset folders, called base/ and cropped/. The base/ and cropped/ folders are both in a folder called lego/. The lego/ folder should be put in a data/ folder in the main project repository, at the same level as src/ and output/. 

## Requirements

This project is coded in Python version 1.109.2. The included setup.sh script creates a virtual environment and installs all dependencies:

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

## Usage

Both scripts can be run from the command line in a bash terminal. Both scripts take two arguments from the command line, data and epochs. 
With the data argument, the user can choose to run the script on the "base" or "cropped" LEGO dataset (base is default). With the epochs argument, the user can choose the number of training iterations (default 10). 

### CNN_direct.py examples

```bash
python CNN_direct.py --data base #runs the script 
#on the base dataset with 10 epochs

python CNN_direct.py --data cropped --epochs 20 #runs 
#the script on the cropped dataset with 20 epochs
```

### CNN_VGG16.py examples

```bash
python CNN_VGG16.py --data base #runs the script 
#on the base dataset with 10 epochs

python CNN_VGG16.py --data cropped --epochs 20 #runs 
#the script on the cropped dataset with 20 epochs
```

## Output

Each run saves two files to the output/ folder, a classification report and a plot of training and validation accuracy and loss. 

Running the CNN_direct.py script saves a report and plot with the names: 

"[dataset]_direct_classification_report.txt" and
"[dataset]_direct_learning_curves.png". 

Running the CNN_VGG16.py script saves a report and plot with the names: 

"[dataset]_VGG16_classification_report.txt" and
"[dataset]_VGG16_learning_curves.png". 

Running both scripts on both datasets will produce 8 output files in total, without any overwriting.

### Key points from the output files 

The CNN_direct.py script had a 50% accuracy on the base dataset and a 64% accuracy on the cropped dataset when run with 10 epochs. This shows that the model can be used to predict about 1 in 2 bricks correctly. That there is a 14 percentage points improvement when using cropped data suggests that the model is not very robust and is affected by background 'noise'. 

The CNN_VGG16.py script performed well, having an 88% accuracy on the base dataset and a 89% accuracy on the cropped dataset when run with 10 epochs. 
That there was almost no difference between the base and cropped datasets for the CNN_VGG16.py script shows that the VGG16 feature extractor is not too affected by backgrounds, but can 'focus' on the brick. 

There was some variability in performance of both models between different bricks. The most intuitively unique looking bricks saw the best performance, such as the "Plate_1x1_Round" (CNN_VGG16.py base f1 score: 1.00) and the "Plate_1x2_Grill" (CNN_VGG16.py base f1 score: 0.99), while the most difficult brick to recognize was the "Brick_1x2" (CNN_VGG16.py base f1 score: 0.73). This is probably due to the similarity between 1x2, 1x3, and 1x4 bricks. 

## Discussion and limitations 

The learning curve plots for CNN_direct.py shows that the model is overfitting, as the training loss is decreasing while validation loss rises. This is not the case for the CNN_VGG16.py plots, which show that both training loss and validation loss are decreasing for both the base and cropped datasets. The plots also show that both models are still learning at epoch 10 and would benefit from running more epochs, as the curves don't flatten out completely. This is more noticable for the CNN_direct.py plot.

These results show that the CNN_VGG16.py model performs on average 25-38 percentage points better than the CNN_direct.py model, depending on the dataset. This illustrates the advantage of using transfer learning to train models for image classification, as a model with a single convulutional layer trained from scratch cannot compete with what e.g. VGG16 has learned from being trained on millions of ImageNet images, especially for a classification task with 20 different labels.

A limitation with using VGG16 is that it is trained on 'natural' images from ImageNet, which do not resemble the LEGO data very much. A general limitation is that the LEGO dataset is not very large, containing a total of 9160 images, which is a modest amount for training models. A potential remedy for this issue is data augmentation, which could increase the amount of data the models have available to train on artificially, which would benefit the CNN_direct.py most. This model would probably also benefit from adding more convolutional layers and adding max pooling to reduce overfitting.
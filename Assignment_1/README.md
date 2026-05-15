# Assignment 1: image search with histograms and image embeddings

### Christian K. Kjeldsen

This project repository contains two python scripts that does image search by comparing images. The first script, flower_histograms.py, does this by extracting a colour histogram of the target image and compares it to the colour histograms of all other images in the dataset. The second script, flowers_cosine.py, extracts image embeddings using VGG16 and calculates cosine similarities between the target image and all other images. In this way, the scripts can be used to search for images that are similar to a target image. Therefore, the performance of these methods can be compared. 

## Repo structure 

```
Assignment_1/
├── data/
│   └── image_0001.jpg ... image_1308.jpg  (1308 images)
├── output/
│   ├── image_0107_cosine_similarity_comparison_results.csv
│   └── image_0107_histogram_comparison_results.csv
└── src/
    ├── flower_cosine.py
    ├── flower_histograms.py
└── requirements.txt
└── setup.sh
└── README.md

```

## Data

The data used in this project is available from:

https://www.robots.ox.ac.uk/~vgg/data/flowers/17/

The dataset contains 1360 image files, with 80 images of 17 different flowers. The images come in a jpg/ folder when downloaded, but should be placed directly in the data/ folder (without the jpg/ folder) in the main project repository. The data/ folder should be at the same level as the src/ and output/ folders. 

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

Both scripts can be run from the command line in a bash terminal. First change directory to the src/ folder. Both scripts take one argument from the command line, the target flower image that one wants to find similar images to. The default image in both scripts if no target image is passed as input is image_0107.jpg. 

### flower_histograms.py examples

```bash
python flower_histograms.py #runs the script on image_0107.jpg
python flower_histograms.py --input image_0500  #runs the 
#script on image_0500.jpg
```

### flower_cosine.py examples

```bash
python flower_cosine.py #runs the script on image_0107.jpg
python flower_cosine.py --input image_0500 #runs the 
#script on image_0500.jpg
```

## Output

Each run saves a single .csv file to the output/ folder containing the five most similar images and their computed values. 

Running the flower_histograms.py script saves a .csv with the names of the 5 images with the lowest differences in histogram values compared to the target image along with their computed distance. Running the flower_cosine.py script saves a .csv with the names of the 5 images with the highest cosine similarities to the target image, along with their computed values. 

### Key points from the output files

Because the two scripts compute different metrics, their returned values cannot be compared directly, but must be compared to a qualitative reading of the returned images. 

image_0107 appears to be a snowdrop. This flower has a green stem with white leaves, and in the background are more examples of snowdrops, yielding an image that is predominantly green and white, with some black spots in the shadows. 

The flower_histogram.py script computes the most similar image to be image_0610, which is a wild tulip, a yellow flower with brown stem standing in front of a brown house and grey sky. The second-most similar image according to the histogram comparison is image_0677, a purple fritillary on a uniform green background. 

The flower_cosine.py script computes the most similar image to be image_0092, which shows 3 sneedrops on a uniform green background. In fact, all 5 returned images are images of snowdrops on green backgrounds. 

## Discussion and limitations 

Based on a qualitative reading of the returned images, it can be concluded that the flower_cosine.py script performs much better than the flower_histogram.py script, as it only returns images of the same species and which are all visually similar. Images computed to have similar histograms do not resemble the target image, showing flowers of different shapes and colours. 

A reason for the poor performance of the histogram method is that a single histogram is computed across all three colour channels at the same time, which might muddy the results, compared to computing a single one for each colour channel separately. Computing colour histograms also ignores spatial information in the images, looking only at colours, while ignoring shape entirely. Intuitively, we know this to be counterproductive to classify flowers. 

The flower_cosine.py script works very well because the VGG16 neural network has been trained to detect shapes and objects. This means that two images of the same flower will have similar embeddings regardless of the lighting, called semantic similarity. This is entirely opposite to the histogram method, which is sensitive to lighting as a change in brightness shifts pixel values, but not to shapes. 
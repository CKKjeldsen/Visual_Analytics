# Assignment 4: Detecting faces in historical newspapers

### Christian K. Kjeldsen

This project repository contains a python script that detects faces in images of historical newspaper pages. The script prints CSV files with the total number of faces per decade and the percentage of pages for that decade which had faces on them for each of the three newspapers. It also prints a plot that shows the diachronic change visually. 

## Repo structure

```
Assignment_4/
├── data/
│   └── newspapers/
│       ├── GDL/
│       │   └── [GDL-YYYY-MM-DD-a-p000X.jpg ...] (1982 files)
│       ├── IMP/
│       │   └── [IMP-YYYY-MM-DD-a-p000X.jpg ...] (1634 files)
│       └── JDG/
│           └── [JDG-YYYY-MM-DD-a-p000X.jpg ...] (1008 files)
├── output/
│   ├── GDL_faces_by_decade.csv
│   ├── GDL_pct_pages_with_faces.png
│   ├── IMP_faces_by_decade.csv
│   ├── IMP_pct_pages_with_faces.png
│   ├── JDG_faces_by_decade.csv
│   └── JDG_pct_pages_with_faces.png
├── src/
│   └── main.py
└── requirements.txt
└── setup.sh
└── README.md
```

## Data

The data used in this project is available from: Barman, R., Ehrmann, M., Clematide, S., Ares Oliveira, S., & Kaplan, F. (2021). Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers. Journal of Data Mining & Digital Humanities, HistoInformatics. https://doi.org/10.5281/zenodo.4065271

The image files are under copyright (property of the journal *Le Temps* and of *ArcInfo*) and can be used for academic research or educational purposes only.

The dataset contains 4624 image files of scanned newspaper pages in three folders named after the newspaper it contains pages from (GDL, IMP, and JDG). All three newspaper subfolders are in a newspaper/ parent folder. A total of 5 files are skipped when running the script because the image file is truncated and so cannot be run. 

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

facenet_pytorch==2.6.0

matplotlib==3.10.9

Pillow==12.2.0

torch==2.2.2

After setup, activate the environment before running any scripts:

```
source ./env/bin/activate
```

## Usage

This script can be run from the command line in a bash terminal. First change directory to the src/ folder. The script does not take any arguments. 

### main.py example

```bash
python main.py
```

## Output

Each run saves two files per newspaper to the output/ folder, a .csv with the number of faces in newspaper pages per decade and a plot showing the percentage of newspaper pages with faces in them per decade. The files are named according to the newspaper it has data from, for example GDL_faces_by_decade.csv and GDL_pct_pages_with_faces for the GDL newspaper. Running the script produces a total of 6 files. 

### Key points from the output files 

From the three figures, we see that the presence of faces in Swiss newspapers generally increased in the period under review, roughly 1800-2000. 

For the GDL paper, the presence seems to have been relatively high in the beginning of the period with faces on about 20% of pages, to decrease in the middle, and then to increase towards the end of the period to have faces on about 35% of pages. 

For the IMP paper, which is the youngest of the three newspapers, the presence of faces seems to have been relatively high all throughout the period, but noticably increasing towards about a 75% presence in the last decades under review. 

For the JDG paper, which had a relatively low presence throughout the entire 19th century, a shift around the year 1900 seems to have occured. From then onwards, the presence of faces seems to have been subtly increasing up to around 30% of newspaper pages. 

In summary, all the three newspapers show an increase in the percentage of pages with faces on them throughout the last two centuries, probably reflecting both advances in printing technology, and later the shift to digital production, as well as changed editorial policy.

## Discussion and limitations 

These results show that it is possible to use visual analytic tools to detect faces in historical newspapers using MTCNN. MTCNN was, however, trained on modern pictures of faces, for which reason it might underperform on dated, low resolution historic images, which would result in the number of faces in newspapers in earlier years in the study being underestimated. Another possible limitation with using MTCNN is that it expects (modern) colour images, while the newspapers are in greyscale. In order to compute, they are turned to RGB, but they are not natively so, which might affect MTCNN's performance. That the dataset is sampled unevenly across years, with some decades having fewer images than others, can also cause irregular deviations in percentages. Further study should sample evenly across years and run the analysis again. 
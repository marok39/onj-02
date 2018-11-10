# Assignment 1: Basic text processing

## Dataset
Dataset was obtained on kaggle: [380,000+ lyrics from MetroLyrics dataset](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics).

## Repo structure

- /input (data input files for Model)
- /report (pdf report with .tex file)
- /results (additional resources)

Inside _/input_ you can find preprocessed input files. These files can be used in Model class to train model and plot results.

 

## How to run

### Run preprocessing script

`python preprocessing_data.py`

If you want to change year selection edit preprocessing_data.py file and update YEAR variable to desired year. Program will select all songs that we're released after that year.

You can also add additional genres to selection by adding them to __PRESELECTED_GENRES__ array.

### Run model script

This script will train a model and plot results.

`python model.py`

If you edited preprocessing_data.py file change file name in model.py to include new data. By default it uses _lyrics_clean_2015.csv_ file


# Restaurant Reviews NLP

## Overview

This project aims to build a model to predict whether a review for a restaurant is considered positive(1) or negative(0). 

General steps to build a model:
- Importing Dataset
- Preprocessing Dataset
- Vectorization
- Training and Classification
- Accuracy Report

## Dataset

The source of the Restaurant Reviews dataset can be downloaded from https://www.kaggle.com/datasets/hj5992/restaurantreviews?resource=download

Data description:
- Num of rows : 1000
- Num of columns : 2
- Column 'Review' data type : string
- Column 'Liked' data type : integer
- Num of unique values : 977

## How to Run the Project

Download the Restaurant Reviews tsv file from  https://www.kaggle.com/datasets/hj5992/restaurantreviews?resource=download

Put the dataset file inside the project in `data/01_raw/dataset.tsv` (dataset.tsv is just an example of your dataset)

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

After all the dependencies are installed, you can run your Kedro project with:

```
kedro run
```

You can also see the graph visualization of your pipeline. To do that, run:
```
kedro run
```



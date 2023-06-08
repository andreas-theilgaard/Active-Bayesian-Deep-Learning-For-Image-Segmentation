Active Bayesian Deep Learning For Image Segmentation
==============================

[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/ubuntu.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/macos.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/macos.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/windows.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/windows.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/Coverage_Report.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/Coverage_Report.yml)

# Contents
- [Project Setup](#Project-Setup)
- [Adding A New Dataset](#adding-a-new-dataset)
- [Project Organization](#project-organization)

# Project Setup
<details>
<summary>
Project Setup
</summary>

#### Clone repository
```
clone https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation.git
```

#### Create virtual environment (require Python 3.10)
Create virtual environment containing the packages used for this project by the following command:
```
conda env create -f environment.yml
```
</details>


# Adding A New Dataset

<details>
<summary> Adding A New Dataset</summary>

In order to add a new dataset and run the experiments with this dataset do the following:
1. Add the data to the ```data/raw/``` folder with the name of dataset as the folder name and using the same structure as showed below.

------------



    ├── data
      ├── color_mapping
      ├── processed
      └── raw
          ├── DIC_C2DH_Hela
          │   ├── image
          │   └── label
          ├── your dataset       <- Your dataset here
          │   ├── image          <- The images of your dataset
          │   └── label          <- The labels of your dataset
--------

2. Go to ```src/config/``` and add your dataset to the Config class like this
```
class Config:
    datasets = ["warwick",.....,"your_dataset_name"]
    n_classes = {'PhC-C2DH-U373' :8,......,'your_dataset_name':number_of_classes_in_your_dataset}
    title_mapper = {"PhC-C2DH-U373": "PhC-U373",......,"your_dataset_name":"Dataset Title"}
```
The "Dataset Title" is the title you want to be showed on various plot. If you are happy with "your_dataset_name" simply use that.

3. Run the following make command in order to create a new color map instance with your dataset included
```
make colors
```
4. You can now execute the experiments described in ?? using your own dataset by passing "your_dataset_name" to the ```dataset``` flag. If you are interested in binary segmentation, you can enable it by setting the binary flag to true. This will divide the mask into two categories: the background (labeled 0) and everything else (labeled 1).

</details>

# Project Organization

<details>
<summary>
Project Organization
</summary>

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

</details>

# Running Experiments

<details>
<summary>
Running Experiments
</summary>
The various experiments can be executed under ```src/experiments/```.
It is recommended to use a GPU.
</details>

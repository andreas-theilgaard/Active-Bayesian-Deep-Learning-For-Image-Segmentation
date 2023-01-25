Active Bayesian Deep Learning For Image Segmentation
==============================

[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/ubuntu.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/macos.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/macos.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/windows.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/windows.yml)
[![build status](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/Coverage_Report.yml/badge.svg)](https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation/actions/workflows/Coverage_Report.yml)

A short description of the project.

## Project Setup
#### Clone repository
```
clone https://github.com/andreas-theilgaard/Active-Bayesian-Deep-Learning-For-Image-Segmentation.git
```

#### Create virtual environment (require Python 3.10)
Create virtual environment containing the packages used for this project by the following command:
```
conda env create -f environment.yml 
```

Project Organization
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

# Youtube Comments Analyzer (YCA) 

## Introduction 

To build a sentiment analysis tool for social media comments.

Youtube has disabled the dislike button on their platforms and creators are not able to determine the likes and dislikes of their audience. The comment section is the only source of truth that they can rely on, let’s build an analyzer to make this easier for them.

Build a simple analyzer including in-depth analysis of comments including top keywords, spam, and fraud detection. Creators can simply provide a Youtube URL or array of comments and the analyzer will scrape through the comments and give the results.

## Setting up the project

We suggest you create and work out of a virtual environment. You can create a virtual environment using conda by following these steps, but of course you can also use whatever you're familiar with:

```bash
conda create -n envname python=x.x anaconda
conda activate envname
```

## Project Structure

This is a rough outline of the project structure.

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Make this project pip installable with `pip install -e`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



**Note**:- This Project is under development, so README.md is also under development, we hope to make it complete in coming weeks. 



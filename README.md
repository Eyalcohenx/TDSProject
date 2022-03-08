# TDSProject

Tabular Data Science course (89-547, BIU, 2022)

This tool is a two liner Gird searcher for outlier detection parameters based on PyOD library algorithms for tabular data.

## Usage

First, load a dataset and split into train and test.

```python

from source.outliers_detector import OutlierGridSearcher

ogs = OutlierGridSearcher(train_func=train_model_function)
X_train, y_train = load_data_function()
X_filtered, y_filtered = ogs.fit(X_train, y_train)

```


## Installation

```shell

git clone https://github.com/Eyalcohenx/TDSProject
cd TDSProject
pip install -r requirements.txt

```

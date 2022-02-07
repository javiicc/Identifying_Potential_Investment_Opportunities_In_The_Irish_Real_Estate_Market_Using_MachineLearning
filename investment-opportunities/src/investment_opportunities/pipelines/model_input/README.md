# Pipeline model_input

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.17.5`.

## Overview

This pipeline prepares the variables needed to train the models and splits the data.

- Nodes:
  - merge_node
  - select_variables_to_modelize_node
  - split_data_node

## Pipeline inputs

- ads_with_location_features_and_geonames

## Pipeline outputs

- X_train
- X_test
- y_train
- y_test
- X
- y

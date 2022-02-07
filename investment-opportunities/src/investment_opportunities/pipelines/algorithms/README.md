# Pipeline algorithms

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.17.5`.

## Overview

This pipeline does trains several models.

- Nodes:
  - model_input_variables_node
  - add_geonames_node

## Pipeline inputs

- df_no_outliers
- geonames
- X_train
- y_train

## Pipeline outputs

- polyr
- knnr
- dtr
- xgbr
- rfr
- levels_list
- num_features
- cat_features
- estimators_dict

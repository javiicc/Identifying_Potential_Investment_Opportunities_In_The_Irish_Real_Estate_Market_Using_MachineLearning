# Pipeline predict_prices

> *Note:* This is a `README.md` boilerplate generated using `Kedro 0.17.5`.

## Overview

This pipeline predict prices and prepare the data for frontend.

- Nodes:
  - get_predictions_node
  - get_residuals_node
  - add_features_for_frontend_node

## Pipeline inputs

- final_model
- X
- y
- df_no_outliers

## Pipeline outputs

- data_for_frontend

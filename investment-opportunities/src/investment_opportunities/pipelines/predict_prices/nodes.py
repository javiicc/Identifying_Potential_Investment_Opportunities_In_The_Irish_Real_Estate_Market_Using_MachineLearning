"""
This is a boilerplate pipeline 'predict_prices'
generated using Kedro 0.17.5
"""

import pandas as pd
import numpy as np


###########################################################################
# PREDICTIONS TRAINED NODES
###########################################################################


def get_predictions(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Predicts prices and return data with a new predicted_price column.

    Parameters
    ----------
    model
    X
    y

    Returns
    -------
    Returns a DataFrame with the input data and predictions.
    """
    y_pred = np.exp(model.predict(X))
    print('-' * 30, 'PREDICTIONS ---------> DONE!!!!!', '-' * 30)
    data_w_predictions = X.copy()
    # Add a column with the target
    data_w_predictions['actual_price'] = y
    # We convert y_pred to a list to avoid a problem with nan values
    # Add a column with the predicted prices
    data_w_predictions['predicted_price'] = list(y_pred)
    return data_w_predictions


def get_residuals(data_w_residuals: pd.DataFrame) -> pd.DataFrame:
    """Computes the residuals and returns the data with a new column with them.

    Parameters
    ----------
    data_w_residuals

    Returns
    -------
    Returns a DataFrame with the input data and residuals.
    """
    # The residual is what will tell us whether the asset is a potential opportunity
    # The residual is the difference between the predicted price and the actual price
    data_w_residuals['residual'] = (data_w_residuals.predicted_price
                                    - data_w_residuals.actual_price)
    data_w_residuals['res_percentage'] = (data_w_residuals.residual
                                          / data_w_residuals.actual_price)
    return data_w_residuals


def add_features_for_frontend(data_w_residuals: pd.DataFrame,
                              model_input: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    data_w_residuals
    model_input :
        This DataFrame was obtained earlier and we will use it to take some useful
        features for our frontend.

    Returns
    -------
    The DataFrame with all data needed for the Dash application.
    """
    columns_to_merge = model_input['url']
    data_for_frontend = data_w_residuals.merge(columns_to_merge,
                                               left_index=True, right_index=True)
    print('-' * 30, 'DATA READY', '-' * 30)
    return data_for_frontend

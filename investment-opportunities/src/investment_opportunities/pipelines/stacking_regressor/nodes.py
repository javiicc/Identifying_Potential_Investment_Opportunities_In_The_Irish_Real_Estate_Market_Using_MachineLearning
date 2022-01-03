"""
This is a boilerplate pipeline 'stacking_regressor'
generated using Kedro 0.17.5
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor


###########################################################################
# STACKING REGRESSOR BA NODE
###########################################################################


def get_stacking(voting_regressor_BA, xgbr, X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Takes the voting regressor and the xgboost, builds a stacking model with them and a
    linear regression and trains it with the given data.

    Parameters
    ----------
    voting_regressor_BA
    xgbr
    X_train
    y_train

    Returns
    -------
    Stacking model trained.
    """
    # Define the base models
    level0 = list()
    level0.append(('voting_regressor_BA', voting_regressor_BA))
    level0.append(('xgb', xgbr))
    # Define meta learner model
    level1 = LinearRegression()
    # Define the stacking ensemble
    stackingr = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    stackingr.fit(X_train, y_train)
    print('-' * 30, 'STACKING REGRESSOR TRAINED!!', '-' * 30)
    return stackingr
